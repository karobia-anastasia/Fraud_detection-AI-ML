import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.conf import settings
from sklearn.preprocessing import LabelEncoder

# Function to clean the data
def clean_data(df):
    # Handle numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        df[col].fillna(df[col].median(), inplace=True)

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)

        # Limit the number of categories by grouping rare values into 'Other'
        df = reduce_categories(df, col)

    # Handle boolean columns (Fraud detection columns)
    boolean_cols = ['isFraud', 'isFlaggedFraud']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Convert categorical columns to dummy variables (sparse)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)

    return df

# Function to reduce the number of categories in categorical columns
def reduce_categories(df, col, threshold=100):
    """Group infrequent categories into 'Other'."""
    value_counts = df[col].value_counts()
    to_replace = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(to_replace, 'Other')
    return df

# Split the data into training and testing sets
def split_data(df):
    if 'isFraud' not in df.columns:
        raise KeyError("'isFraud' column not found in the dataset")

    X = df.drop('isFraud', axis=1)  # Features
    y = df['isFraud'].astype(int)   # Target (convert to binary if needed)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Train and evaluate the model based on the type (RandomForest, LogisticRegression, XGBoost)
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_type} Model Accuracy: {accuracy * 100:.2f}%")

    return model


# Save the trained model
def save_model(model, model_name='fraud_detection_model'):
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

    # Save the model to a file
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Load the trained model
def load_model(model_name='fraud_detection_model'):
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{model_name}.pkl')

    # Check if the model exists
    if os.path.exists(model_path):
        try:
            # Try to load the model
            model = joblib.load(model_path)
            return model
        except EOFError:
            # If EOFError occurs (corrupted model file), raise an error
            raise EOFError("Model file appears to be corrupted. Please re-train the model.")
    else:
        raise FileNotFoundError("Model not found. Please ensure the model is saved and available.")

# Preprocess the input data for prediction

def align_columns(df, model):
    # Get the features that the model was trained on (stored in the model)
    model_features = model.feature_names_in_  # This retrieves the original feature names from the trained model

    # Add missing columns with default values (e.g., 0)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  # Default value for missing columns
    
    # Remove any columns that the model doesn't expect (e.g., 'isFlaggedFraud' if it's in the input but not in the model)
    df = df[model_features]  # Reorder columns to match the model's expected input order

    return df

def preprocess_for_prediction(data):
    # Assuming `data` is a dictionary with relevant features
    df = pd.DataFrame([data])

    # Handle categorical columns (same as cleaning)
    categorical_cols = ['type', 'nameOrig', 'nameDest']  # Example, replace with your actual categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    # Convert categorical columns to dummy variables (sparse)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)

    return df

# Prediction function
def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest, model_type='RandomForest'):
    # Create a dictionary of the transaction data
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

    # Load the model
    model = load_model(model_name=model_type)

    # Preprocess the data (no scaler needed)
    processed_data = preprocess_for_prediction(transaction_data)

    # Make the prediction
    prediction = model.predict(processed_data)

    # Return the prediction result (fraudulent or not)
    return "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"


import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.core.paginator import Paginator
from .models import Transaction
import json
import base64
from io import BytesIO
# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def prediction_results(request):
    """Display results of model predictions on transaction data."""
    transactions = Transaction.objects.all()
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X = pd.DataFrame(list(transactions.values()))[features]

    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')

    if os.path.exists(model_path):
        # Load the pre-trained model from file
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError('Model file not found. Please train the model first.')

    # Use the model to make predictions on the transaction data
    predictions = model.predict(X)

    # Add predictions and labels to the DataFrame
    transaction_df = pd.DataFrame(list(transactions.values()))
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Update each transaction's prediction in the database
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    # Calculate prediction counts (Fraud vs Non-Fraud)
    prediction_counts = {
        'Non-Fraud': sum([t.prediction == 0 for t in transactions]),
        'Fraud': sum([t.prediction == 1 for t in transactions]),
    }

    # Get true labels from the database (assuming 'isFraud' exists)
    y_true = pd.Series([t.isFraud for t in transactions])  # True labels from database

    # Calculate metrics: accuracy, precision, recall, f1_score, and roc_auc
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    roc_auc = roc_auc_score(y_true, predictions)

    # Prepare report dictionary with the metrics
    report = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    # Generate confusion matrix
    cm = confusion_matrix(y_true, predictions)
    confusion_image = plot_confusion_matrix(cm)

    # Paginate the transactions for display in the template
    paginator = Paginator(transactions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Render the template with the calculated results
    return render(request, 'data_analysis.html', {
        'page_obj': page_obj,
        'report': report,
        'prediction_data': prediction_counts,
        'confusion_image': confusion_image
    })

# Function to plot confusion matrix as an image
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save to in-memory image buffer
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img_to_base64(img)

# Convert image to base64 for embedding in HTML
def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()
# Prediction Reports view
def prediction_reports(request):
    transactions = Transaction.objects.all()
    transaction_df = pd.DataFrame(list(transactions.values()))
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Get feature columns from the DataFrame
    X = transaction_df[features]
    y = transaction_df['isFraud']
    
    # Split the data into training and testing sets (for model re-training or evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    # Train a RandomForestClassifier model with balanced class weights to handle class imbalance
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # Save the trained model to disk
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Use the model to make predictions on all transactions
    predictions = model.predict(X)

    # Add the predictions and prediction labels to the DataFrame
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Update each transaction's prediction in the database
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    # Confusion Matrix Calculation
    cm = confusion_matrix(y, predictions)
    confusion_image = plot_confusion_matrix(cm)  # Generates confusion matrix image
    
    # Prediction Counts
    prediction_counts = pd.Series(predictions).value_counts()
    prediction_chart = plot_prediction_counts(prediction_counts)  # Bar chart of fraud vs non-fraud
    
    # Prepare prediction count data for the template
    prediction_data = {
        'labels': ['Non-Fraud', 'Fraud'],
        'data': [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]
    }
    
    # Classification Report
    report = classification_report(y, predictions, output_dict=True)

    return render(request, 'dashboard_view.html', {
        'transaction_data': transaction_df,
        'report': {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
        },
        'prediction_data': prediction_data,
        'confusion_image': confusion_image,
        'prediction_chart': prediction_chart
    })

# Function to plot confusion matrix as an image
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save to in-memory image buffer
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img_to_base64(img)

# Function to plot prediction counts as a bar chart
def plot_prediction_counts(prediction_counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette='Blues')
    ax.set_title('Prediction Counts: Fraud vs Non-Fraud')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Count')

    # Save to in-memory image buffer
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)

# Convert image to base64 for embedding in HTML
def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')

    # Save to in-memory image buffer
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], align='center')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(np.array(feature_names)[indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance')

    # Save to in-memory image buffer
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)


# Compare multiple models
def compare_models(request):
    """Train and compare multiple models for fraud detection performance."""
    
    # Load the transaction data
    transactions = Transaction.objects.all()
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    transaction_df = pd.DataFrame(list(transactions.values()))
    X = transaction_df[features]
    y = transaction_df['isFraud']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Store metrics for each model
    model_metrics = {}

    for model_name, model in models.items():
        logger.info(f"Training {model_name} model...")
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for ROC curve
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        model_metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

    # Convert metrics to a DataFrame for easy plotting
    metrics_df = pd.DataFrame(model_metrics).T

    # Plot comparison of model metrics
    comparison_image = plot_model_comparison(metrics_df)

    # Render the template with the model comparison results
    return render(request, 'model_comparison.html', {
        'metrics_df': metrics_df,
        'comparison_image': comparison_image
    })

def plot_model_comparison(metrics_df):
    """Plot model comparison metrics (Accuracy, Precision, Recall, F1, ROC-AUC)."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax, colormap='Blues', width=0.8)
    
    # Set labels and title
    ax.set_title('Model Performance Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')

    # Save to in-memory image buffer
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img_to_base64(img)

# Convert image to base64 for embedding in HTML
def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()


