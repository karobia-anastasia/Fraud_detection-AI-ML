import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.conf import settings
import seaborn as sns


# Clean Data Function
def clean_data(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    boolean_cols = ['isFraud', 'isFlaggedFraud']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
    
    return df

# Reduce categories for categorical variables
def reduce_categories(df, col, threshold=100):
    value_counts = df[col].value_counts()
    to_replace = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(to_replace, 'Other')
    return df

# Split data into training, validation, and test sets
def split_data(df):
    if 'isFraud' not in df.columns:
        raise KeyError("'isFraud' column not found in the dataset")
    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Perform grid search for hyperparameter tuning
def perform_hyperparameter_tuning(X_train, y_train, model_type='RandomForest', search_type='GridSearch'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
    elif model_type == 'XGBoost':
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 6, 10, 12],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.3]
        }
    else:
        raise ValueError("Unsupported model type")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if search_type == 'GridSearch':
        search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2)
        print("Performing Grid Search...")
    elif search_type == 'RandomizedSearch':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, scoring='accuracy', 
                                    cv=cv, n_jobs=-1, verbose=2, random_state=42)
        print("Performing Randomized Search...")
    else:
        raise ValueError("Unsupported search type. Choose 'GridSearch' or 'RandomizedSearch'.")

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    return best_model, best_params

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, search_type='GridSearch'):
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    base_predictions = base_model.predict(X_test)
    base_accuracy = accuracy_score(y_test, base_predictions)
    
    print(f"Hyperparameter tuning for XGBoost using {search_type}...")
    xgb_model, best_xgb_params = perform_hyperparameter_tuning(X_train, y_train, model_type='XGBoost', search_type=search_type)

    xgb_model.fit(X_train, y_train)
    
    xgb_test_predictions = xgb_model.predict(X_test)
    xgb_test_accuracy = accuracy_score(y_test, xgb_test_predictions)
    xgb_precision = precision_score(y_test, xgb_test_predictions, average='weighted')
    xgb_recall = recall_score(y_test, xgb_test_predictions, average='weighted')
    xgb_f1 = f1_score(y_test, xgb_test_predictions, average='weighted')

    xgb_improvement = ((xgb_test_accuracy - base_accuracy) / base_accuracy) * 100

    print(f"Base Model (RandomForest) Accuracy: {base_accuracy:.4f}")
    print(f"Improved Model (XGBoost) Accuracy: {xgb_test_accuracy:.4f}")
    print(f"XGBoost Precision: {xgb_precision:.4f}")
    print(f"XGBoost Recall: {xgb_recall:.4f}")
    print(f"XGBoost F1-Score: {xgb_f1:.4f}")
    print(f"XGBoost Accuracy Improvement: {xgb_improvement:.2f}%")
    print(f"Best XGBoost Hyperparameters: {best_xgb_params}")

    plot_comparison_graph(base_accuracy, xgb_test_accuracy, xgb_improvement)

    return base_model, xgb_model, xgb_improvement


def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Function to plot accuracy comparison
def plot_comparison_graph(base_accuracy, xgb_accuracy, improvement):
    plt.figure(figsize=(8, 6))
    plt.bar(['RandomForest', 'XGBoost'], [base_accuracy, xgb_accuracy], color=['blue', 'orange'])
    plt.title(f'Accuracy Comparison (Improvement: {improvement:.2f}%)')
    plt.ylabel('Accuracy')
    
   
    img = BytesIO()
    plt.savefig(img, format='png')  # Save before calling plt.show()
    img.seek(0)  # Rewind the buffer
    plt.close()  # Close the figure to avoid memory leaks
    
    # Convert to base64 string
    img_base64 = img_to_base64(img)
    return img_base64

# Save model to disk
def save_model(model, model_name='fraud_detection_model'):
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Load model from disk
def load_model(model_name='fraud_detection_model'):
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except EOFError:
            raise EOFError("Model file appears to be corrupted. Please re-train the model.")
    else:
        raise FileNotFoundError("Model not found. Please ensure the model is saved and available.")


def align_columns(df, model):
    expected_features = model.get_booster().feature_names if hasattr(model, 'get_booster') else model.feature_names_in_
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    aligned_df = df[expected_features]
    return aligned_df


# Preprocess input data for prediction
def preprocess_for_prediction(data):
    df = pd.DataFrame([data])

    categorical_cols = ['type', 'nameOrig', 'nameDest'] 
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
    return df

# Predict fraud status for new transactions
def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest, model_type='RandomForest'):
    transaction_data = {
        'step': step,
        'type': transaction_type,
        'amount': amount,
        'nameOrig': name_orig,
        'oldbalanceOrg': old_balance_orig,
        'newbalanceOrig': new_balance_orig,
        'nameDest': name_dest,
        'oldbalanceDest': old_balance_dest,
        'newbalanceDest': new_balance_dest
    }
    
    df = preprocess_for_prediction(transaction_data)

    model = load_model(model_type)
    if model is not None:
        df = align_columns(df, model)
        prediction = model.predict(df)[0]

        return "Fraud" if prediction == 1 else "Not Fraud"
    else:
        raise ValueError(f"Model '{model_type}' not found!")
