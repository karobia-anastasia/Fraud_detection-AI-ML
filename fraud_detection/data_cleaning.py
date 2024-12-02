import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.conf import settings

# Function to clean data
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

def reduce_categories(df, col, threshold=100):
    value_counts = df[col].value_counts()
    to_replace = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(to_replace, 'Other')
    return df

def split_data(df):
    if 'isFraud' not in df.columns:
        raise KeyError("'isFraud' column not found in the dataset")
    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].astype(int)  

    return train_test_split(X, y, test_size=0.2, random_state=42)

def perform_grid_search(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
    elif model_type == 'XGBoost':
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    else:
        raise ValueError("Unsupported model type")

    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best Hyperparameters for {model_type}: {best_params}")  # Prints the best parameters
    
    return best_model, best_params

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Base model: RandomForest (No grid search)
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    base_predictions = base_model.predict(X_test)
    base_accuracy = accuracy_score(y_test, base_predictions)
    base_precision = precision_score(y_test, base_predictions)
    base_recall = recall_score(y_test, base_predictions)
    base_f1 = f1_score(y_test, base_predictions)

    # Log base model hyperparameters
    print(f"Base Model (RandomForest) Hyperparameters: {base_model.get_params()}")

    # Perform Grid Search for XGBoost
    xgb_model, best_xgb_params = perform_grid_search(X_train, y_train, model_type='XGBoost')

    # Train the XGBoost model with the best hyperparameters found
    xgb_model.fit(X_train, y_train)
    
    # Evaluate the XGBoost model
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_precision = precision_score(y_test, xgb_predictions)
    xgb_recall = recall_score(y_test, xgb_predictions)
    xgb_f1 = f1_score(y_test, xgb_predictions)
    
    # Calculate percentage improvement
    improvement = ((xgb_accuracy - base_accuracy) / base_accuracy) * 100

    # Log the results
    print("Base Model (RandomForest) - Accuracy:", base_accuracy)
    print("Base Model (RandomForest) - Precision:", base_precision)
    print("Base Model (RandomForest) - Recall:", base_recall)
    print("Base Model (RandomForest) - F1 Score:", base_f1)

    print("Improved Model (XGBoost) - Accuracy:", xgb_accuracy)
    print("Improved Model (XGBoost) - Precision:", xgb_precision)
    print("Improved Model (XGBoost) - Recall:", xgb_recall)
    print("Improved Model (XGBoost) - F1 Score:", xgb_f1)

    print(f"Percentage Improvement (Accuracy): {improvement:.2f}%")

    return base_model, xgb_model, improvement

# Function to plot comparison graph
def plot_comparison_graph(base_accuracy, xgb_accuracy, improvement):
    models = ['Random Forest', 'XGBoost']
    accuracies = [base_accuracy, xgb_accuracy]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies, palette='Blues')
    plt.title(f"Model Comparison\nAccuracy Improvement: {improvement:.2f}%")
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img_to_base64(img)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_type} Model Accuracy: {accuracy * 100:.2f}%")

    return model

def save_model(model, model_name='fraud_detection_model'):
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

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
    model_features = model.feature_names_in_ 
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  
    
    df = df[model_features] 
    return df

def preprocess_for_prediction(data):
    df = pd.DataFrame([data])

    categorical_cols = ['type', 'nameOrig', 'nameDest'] 
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
    return df

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

def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()
