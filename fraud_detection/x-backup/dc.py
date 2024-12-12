# import os
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# from django.conf import settings

# def clean_data(df):
#     numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     print("Numerical Columns before cleaning:", numerical_cols)  # Debugging line
#     for col in numerical_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce') 
#         df[col].fillna(df[col].median(), inplace=True)

#     categorical_cols = df.select_dtypes(include=['object']).columns
#     print("Categorical Columns before cleaning:", categorical_cols)  # Debugging line
#     for col in categorical_cols:
#         df[col] = df[col].astype('category')
#         df[col].fillna(df[col].mode()[0], inplace=True)
#         df = reduce_categories(df, col)

#     boolean_cols = ['isFraud', 'isFlaggedFraud']
#     for col in boolean_cols:
#         if col in df.columns:
#             df[col] = df[col].astype(int)

#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
    
#     print("DataFrame after cleaning:", df.columns)  # Debugging line
#     return df

# def reduce_categories(df, col, threshold=100):
#     value_counts = df[col].value_counts()
#     to_replace = value_counts[value_counts < threshold].index
#     df[col] = df[col].replace(to_replace, 'Other')
#     return df

# def split_data(df):
#     if 'isFraud' not in df.columns:
#         raise KeyError("'isFraud' column not found in the dataset")
#     X = df.drop('isFraud', axis=1)
#     y = df['isFraud'].astype(int)

#     # Split the data into training (60%), validation (20%), and test (20%) sets
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     return X_train, X_val, X_test, y_train, y_val, y_test

# def perform_grid_search(X_train, y_train, model_type='RandomForest'):
#     if model_type == 'RandomForest':
#         model = RandomForestClassifier(random_state=42)
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [10, 20, 30],
#             'min_samples_split': [2, 5, 10],
#             'class_weight': ['balanced', None]
#         }
#     elif model_type == 'XGBoost':
#         model = XGBClassifier(random_state=42)
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [3, 6, 10],
#             'learning_rate': [0.01, 0.1, 0.3],
#             'subsample': [0.8, 1.0]
#         }
#     else:
#         raise ValueError("Unsupported model type")

#     # Grid search (without cross-validation)
#     best_score = -1
#     best_model = None
#     best_params = {}

#     for n_estimators in param_grid.get('n_estimators', [100]):
#         for max_depth in param_grid.get('max_depth', [None]):
#             for min_samples_split in param_grid.get('min_samples_split', [2]):
#                 for class_weight in param_grid.get('class_weight', [None]):
#                     for C in param_grid.get('C', [1]):
#                         for solver in param_grid.get('solver', ['liblinear']):
#                             for max_iter in param_grid.get('max_iter', [100]):
#                                 # Set model parameters
#                                 model.set_params(
#                                     n_estimators=n_estimators,
#                                     max_depth=max_depth,
#                                     min_samples_split=min_samples_split,
#                                     class_weight=class_weight
#                                 )

#                                 # Fit the model and evaluate on the training set
#                                 model.fit(X_train, y_train)
#                                 score = model.score(X_train, y_train)

#                                 # Track the best model
#                                 if score > best_score:
#                                     best_score = score
#                                     best_model = model
#                                     best_params = model.get_params()

#     print(f"Best Hyperparameters for {model_type}: {best_params}")
#     return best_model, best_params

# def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
#     base_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     base_model.fit(X_train, y_train)
    
#     # Evaluate the base model
#     base_predictions = base_model.predict(X_test)
#     base_accuracy = accuracy_score(y_test, base_predictions)
#     base_precision = precision_score(y_test, base_predictions)
#     base_recall = recall_score(y_test, base_predictions)
#     base_f1 = f1_score(y_test, base_predictions)

#     # Log base model hyperparameters
#     print(f"Base Model (RandomForest) Hyperparameters: {base_model.get_params()}")

#     # Perform Grid Search for XGBoost (removed Logistic Regression)
#     xgb_model, best_xgb_params = perform_grid_search(X_train, y_train, model_type='XGBoost')

#     # Train the XGBoost model with the best hyperparameters found
#     xgb_model.fit(X_train, y_train)

#     # Evaluate the XGBoost model on the test set
#     xgb_test_predictions = xgb_model.predict(X_test)
#     xgb_test_accuracy = accuracy_score(y_test, xgb_test_predictions)
#     xgb_test_precision = precision_score(y_test, xgb_test_predictions)
#     xgb_test_recall = recall_score(y_test, xgb_test_predictions)
#     xgb_test_f1 = f1_score(y_test, xgb_test_predictions)
#     xgb_test_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])  # AUC

#     # Calculate percentage improvement for XGBoost
#     xgb_improvement = ((xgb_test_accuracy - base_accuracy) / base_accuracy) * 100

#     # Log the results
#     print("Base Model (RandomForest) - Accuracy:", base_accuracy)
#     print("Improved Model (XGBoost) - Accuracy:", xgb_test_accuracy)
#     print(f"Percentage Improvement (XGBoost Accuracy): {xgb_improvement:.2f}%")

#     # Plot comparison graph with improvement

#     return base_model, xgb_model, xgb_improvement

# def save_model(model, model_name='fraud_detection_model'):
#     model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")

# def load_model(model_name='fraud_detection_model'):
#     model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

#     if os.path.exists(model_path):
#         try:
#             model = joblib.load(model_path)
#             return model
#         except EOFError:
#             raise EOFError("Model file appears to be corrupted. Please re-train the model.")
#     else:
#         raise FileNotFoundError("Model not found. Please ensure the model is saved and available.")

# def align_columns(df, model):
#     model_features = model.feature_names_in_ 
#     for col in model_features:
#         if col not in df.columns:
#             df[col] = 0  
#     df = df[model_features] 
#     return df

# def preprocess_for_prediction(data):
#     df = pd.DataFrame([data])

#     categorical_cols = ['type', 'nameOrig', 'nameDest'] 
#     for col in categorical_cols:
#         df[col] = df[col].astype('category')
#         df[col].fillna(df[col].mode()[0], inplace=True)
#         df = reduce_categories(df, col)

#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
#     return df

# def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest, model_type='RandomForest'):
#     transaction_data = {
#         'step': step,
#         'type': transaction_type,
#         'amount': amount,
#         'nameOrig': name_orig,
#         'oldbalanceOrg': old_balance_orig,
#         'newbalanceOrig': new_balance_orig,
#         'nameDest': name_dest,
#         'oldbalanceDest': old_balance_dest,
#         'newbalanceDest': new_balance_dest
#     }
    
#     df = preprocess_for_prediction(transaction_data)

#     model = load_model(model_type)
#     if model is not None:
#         df = align_columns(df, model)
#         prediction = model.predict(df)[0]

#         return "Fraud" if prediction == 1 else "Not Fraud"
#     else:
#         raise ValueError(f"Model '{model_type}' not found!")

# def img_to_base64(img):
#     return base64.b64encode(img.getvalue()).decode()




# import os
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# from django.conf import settings
# import seaborn as sns

# # Clean Data Function
# def clean_data(df):
#     numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     for col in numerical_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce') 
#         df[col].fillna(df[col].median(), inplace=True)

#     categorical_cols = df.select_dtypes(include=['object']).columns
#     for col in categorical_cols:
#         df[col] = df[col].astype('category')
#         df[col].fillna(df[col].mode()[0], inplace=True)
#         df = reduce_categories(df, col)

#     boolean_cols = ['isFraud', 'isFlaggedFraud']
#     for col in boolean_cols:
#         if col in df.columns:
#             df[col] = df[col].astype(int)

#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
    
#     return df

# # Reduce categories for categorical variables
# def reduce_categories(df, col, threshold=100):
#     value_counts = df[col].value_counts()
#     to_replace = value_counts[value_counts < threshold].index
#     df[col] = df[col].replace(to_replace, 'Other')
#     return df

# # Split data into training, validation, and test sets
# def split_data(df):
#     if 'isFraud' not in df.columns:
#         raise KeyError("'isFraud' column not found in the dataset")
#     X = df.drop('isFraud', axis=1)
#     y = df['isFraud'].astype(int)

#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     return X_train, X_val, X_test, y_train, y_val, y_test

# # Perform grid search for hyperparameter tuning
# def perform_grid_search(X_train, y_train, model_type='RandomForest'):
#     if model_type == 'RandomForest':
#         model = RandomForestClassifier(random_state=42)
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [10, 20, 30],
#             'min_samples_split': [2, 5, 10],
#             'class_weight': ['balanced', None]
#         }
#     elif model_type == 'XGBoost':
#         model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [3, 6, 10],
#             'learning_rate': [0.01, 0.1, 0.3],
#             'subsample': [0.8, 1.0],
#             'colsample_bytree': [0.8, 1.0]
#         }
#     else:
#         raise ValueError("Unsupported model type")

#     # Use GridSearchCV to search for the best hyperparameters with cross-validation
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
#     grid_search.fit(X_train, y_train)

#     # Get the best model and hyperparameters
#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_

#     return best_model, best_params

# # Train and evaluate base (RandomForest) and proposed (XGBoost) models
# def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
#     # Train base model (Random Forest)
#     base_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     base_model.fit(X_train, y_train)
    
#     # Evaluate base model
#     base_predictions = base_model.predict(X_test)
#     base_accuracy = accuracy_score(y_test, base_predictions)
    
#     # Perform Grid Search for XGBoost
#     xgb_model, best_xgb_params = perform_grid_search(X_train, y_train, model_type='XGBoost')

#     # Train the XGBoost model with the best hyperparameters found
#     xgb_model.fit(X_train, y_train)

#     # Evaluate XGBoost model
#     xgb_test_predictions = xgb_model.predict(X_test)
#     xgb_test_accuracy = accuracy_score(y_test, xgb_test_predictions)

#     # Calculate percentage improvement for XGBoost
#     xgb_improvement = ((xgb_test_accuracy - base_accuracy) / base_accuracy) * 100

#     # Log results
#     print(f"Base Model (RandomForest) Accuracy: {base_accuracy}")
#     print(f"Improved Model (XGBoost) Accuracy: {xgb_test_accuracy}")
#     print(f"XGBoost Accuracy Improvement: {xgb_improvement:.2f}%")

#     # Plot comparison graph
#     plot_comparison_graph(base_accuracy, xgb_test_accuracy, xgb_improvement)

#     return base_model, xgb_model, xgb_improvement

# # Plot comparison graph for model performance
# def plot_comparison_graph(base_accuracy, xgb_accuracy, improvement):
#     models = ['Random Forest', 'XGBoost']
#     accuracies = [base_accuracy, xgb_accuracy]

#     # Create a bar plot comparing accuracies of both models
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=models, y=accuracies, palette='Blues')
#     plt.title(f"Model Comparison\nAccuracy Improvement: {improvement:.2f}%")
#     plt.xlabel('Model')
#     plt.ylabel('Accuracy')

#     # Save the plot to a base64 encoded string to use in the web interface (e.g., Django)
#     img = BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plt.close()

#     # Convert to base64 string to send it to the web frontend
#     img_base64 = img_to_base64(img)
#     return img_base64

# # Convert image to base64
# def img_to_base64(img):
#     return base64.b64encode(img.getvalue()).decode()

# # Save model to disk
# def save_model(model, model_name='fraud_detection_model'):
#     model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")

# # Load model from disk
# def load_model(model_name='fraud_detection_model'):
#     model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

#     if os.path.exists(model_path):
#         try:
#             model = joblib.load(model_path)
#             return model
#         except EOFError:
#             raise EOFError("Model file appears to be corrupted. Please re-train the model.")
#     else:
#         raise FileNotFoundError("Model not found. Please ensure the model is saved and available.")

# # Align the input data columns with the model's features
# def align_columns(df, model):
#     model_features = model.feature_names_in_ 
#     for col in model_features:
#         if col not in df.columns:
#             df[col] = 0  
#     df = df[model_features] 
#     return df

# # Preprocess input data for prediction
# def preprocess_for_prediction(data):
#     df = pd.DataFrame([data])

#     categorical_cols = ['type', 'nameOrig', 'nameDest'] 
#     for col in categorical_cols:
#         df[col] = df[col].astype('category')
#         df[col].fillna(df[col].mode()[0], inplace=True)
#         df = reduce_categories(df, col)

#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
#     return df

# # Predict fraud status for new transactions
# def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest, model_type='RandomForest'):
#     transaction_data = {
#         'step': step,
#         'type': transaction_type,
#         'amount': amount,
#         'nameOrig': name_orig,
#         'oldbalanceOrg': old_balance_orig,
#         'newbalanceOrig': new_balance_orig,
#         'nameDest': name_dest,
#         'oldbalanceDest': old_balance_dest,
#         'newbalanceDest': new_balance_dest
#     }
    
#     df = preprocess_for_prediction(transaction_data)

#     model = load_model(model_type)
#     if model is not None:
#         df = align_columns(df, model)
#         prediction = model.predict(df)[0]

#         return "Fraud" if prediction == 1 else "Not Fraud"
#     else:
#         raise ValueError(f"Model '{model_type}' not found!")


import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    else:
        raise ValueError("Unsupported model type")

    # Use GridSearchCV to search for the best hyperparameters with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model and hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

# Train and evaluate base (RandomForest) and proposed (XGBoost) models
def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    # Train base model (Random Forest)
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Evaluate base model
    base_predictions = base_model.predict(X_test)
    base_accuracy = accuracy_score(y_test, base_predictions)
    
    # Perform Grid Search for XGBoost
    xgb_model, best_xgb_params = perform_grid_search(X_train, y_train, model_type='XGBoost')

    # Train the XGBoost model with the best hyperparameters found
    xgb_model.fit(X_train, y_train)

    # Evaluate XGBoost model
    xgb_test_predictions = xgb_model.predict(X_test)
    xgb_test_accuracy = accuracy_score(y_test, xgb_test_predictions)

    # Calculate percentage improvement for XGBoost
    xgb_improvement = ((xgb_test_accuracy - base_accuracy) / base_accuracy) * 100

    # Log results
    print(f"Base Model (RandomForest) Accuracy: {base_accuracy}")
    print(f"Improved Model (XGBoost) Accuracy: {xgb_test_accuracy}")
    print(f"XGBoost Accuracy Improvement: {xgb_improvement:.2f}%")

    # Plot comparison graph
    plot_comparison_graph(base_accuracy, xgb_test_accuracy, xgb_improvement)

    return base_model, xgb_model, xgb_improvement

# Plot comparison graph for model performance
def plot_comparison_graph(base_accuracy, xgb_accuracy, improvement):
    models = ['Random Forest', 'XGBoost']
    accuracies = [base_accuracy, xgb_accuracy]

    # Create a bar plot comparing accuracies of both models
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies, palette='Blues')
    plt.title(f"Model Comparison\nAccuracy Improvement: {improvement:.2f}%")
    plt.xlabel('Model')
    plt.ylabel('Accuracy')

    # Save the plot to a base64 encoded string to use in the web interface (e.g., Django)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert to base64 string to send it to the web frontend
    img_base64 = img_to_base64(img)
    return img_base64

# Convert image to base64
def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()

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

# Align the input data columns with the model's features
def align_columns(df, model):
    model_features = model.feature_names_in_ 
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  
    df = df[model_features] 
    return df

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
