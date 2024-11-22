import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from django.conf import settings
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

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

    # Perform one-hot encoding only on categorical columns
    df = pd.get_dummies(df, drop_first=True, sparse=True)

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

def train_and_evaluate_model(transaction_df):
    # Clean the data first
    cleaned_df = clean_data(transaction_df)

    # Define features and target
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Check if all required columns are present
    missing_cols = [col for col in features if col not in cleaned_df.columns]
    if missing_cols:
        raise KeyError(f"The following columns are missing from the DataFrame: {missing_cols}")
    
    X = cleaned_df[features]
    y = cleaned_df['isFraud']  # Assuming 'isFraud' is the target column

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LogisticRegression': LogisticRegression(solver='liblinear', class_weight='balanced')
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ''
    model_performance = {}

    # Track the performance of each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy, precision, recall, and f1-score for each model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        model_performance[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Determine the best model based on accuracy (can also use f1_score, precision, recall, etc.)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = model

    # Save the best model
    save_model(best_model)

    return model_performance, best_model_name, best_accuracy


def save_model(model):
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    joblib.dump(model, model_path)

def load_model():
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def preprocess_for_prediction(data):
    df = pd.DataFrame([data])

    categorical_cols = ['type', 'nameOrig', 'nameDest']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)

    return df
def align_columns(df, model):
    # Ensure that the input data has the same columns as the model's expected features
    model_features = model.feature_names_in_

    # Add missing columns with default value 0 (since these features were not in the input data)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match the model's training order
    df = df[model_features]

    return df


def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest):
    # Prepare data for prediction
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

    # Load the best saved model
    model = load_model()

    # Preprocess the transaction data
    processed_data = preprocess_for_prediction(transaction_data)

    # Align columns to match the model's expected features
    processed_data = align_columns(processed_data, model)

    # Predict using the best model
    prediction = model.predict(processed_data)

    return "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"
