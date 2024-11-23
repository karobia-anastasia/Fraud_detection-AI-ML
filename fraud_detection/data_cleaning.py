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
import logging

# Set up logging
logger = logging.getLogger(__name__)

def clean_data(df):
    """Clean the input data."""
    # Ensure that the numerical columns are properly converted to float64
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    for col in numerical_cols:
        # Convert columns to numeric, coerce any errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle any NaN values by filling with the median of the column
        df[col].fillna(df[col].median(), inplace=True)

    # Convert other numerical columns (float or int) if they exist in the dataset
    df[numerical_cols] = df[numerical_cols].astype('float64')

    # Convert categorical columns to 'category' type and handle missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    # Perform one-hot encoding only on categorical columns
    df = pd.get_dummies(df, drop_first=True, sparse=True)

    return df

def reduce_categories(df, col, threshold=100):
    """Reduce categories in categorical columns that occur less than a threshold."""
    value_counts = df[col].value_counts()
    to_replace = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(to_replace, 'Other')
    return df

def split_data(df):
    """Split the data into features and target."""
    if 'isFraud' not in df.columns:
        raise KeyError("'isFraud' column not found in the dataset")

    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_comparison_bar_chart(model_performance):
    """Generate a bar chart comparing model accuracies."""
    # Log the model performance to verify the data
    logger.debug(f"Model Performance for plotting: {model_performance}")
    
    # Prepare data for plotting
    model_names = list(model_performance.keys())
    accuracies = [model['accuracy'] for model in model_performance.values()]
    
    # Check if accuracies list is populated
    if not accuracies:
        logger.warning("No accuracy values available for plotting.")
        return None  # Return None if there is no data to plot

    # Plot accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=model_names, y=accuracies, palette='Blues', ax=ax)
    ax.set_title('Model Accuracy Comparison: Random Forest vs Logistic Regression vs XGBoost', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)

    # Save the plot as a PNG image in memory (in a buffer)
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    # Convert the image buffer to base64 and return
    return img_to_base64(img_buffer)

def img_to_base64(img_buffer):
    """Helper function to convert image buffer to base64."""
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

def train_and_evaluate_model(transaction_df):
    """Train and evaluate models, comparing Random Forest with Logistic Regression and XGBoost."""
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
            'accuracy': accuracy,  # Store raw accuracy value
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

    # Log the model performance to ensure it's populated
    logger.debug(f"Model Performance: {model_performance}")

    # Plot the performance comparison bar chart
    accuracy_comparison_img = plot_comparison_bar_chart(model_performance)

    # Output comparison and performance improvement details
    comparison_details = f"""
    5.7.5 Comparison with Baseline Models and Performance Improvement%

    To evaluate the performance of the Random Forest model in the context of fraud detection, it was compared with two baseline models: Logistic Regression and XGBoost.
    - Logistic Regression is a linear model frequently used for binary classification tasks, including fraud detection.
    - XGBoost is an advanced gradient-boosting model that leverages multiple decision trees to improve prediction accuracy and handle complex patterns more effectively.

    Model Evaluation Results:

    Performance Metrics:
    - Random Forest: {model_performance['RandomForest']['accuracy'] * 100:.2f}% accuracy
    - Logistic Regression: {model_performance['LogisticRegression']['accuracy'] * 100:.2f}% accuracy
    - XGBoost: {model_performance['XGBoost']['accuracy'] * 100:.2f}% accuracy

    Visualization: Accuracy Comparison Bar Chart

    Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy
    """

    return model_performance, best_model_name, best_accuracy, accuracy_comparison_img, comparison_details

def save_model(model):
    """Save the trained model to disk."""
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    joblib.dump(model, model_path)

def load_model():
    """Load the trained model from disk."""
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def preprocess_for_prediction(data):
    """Preprocess input data for making predictions."""
    df = pd.DataFrame([data])

    categorical_cols = ['type', 'nameOrig', 'nameDest']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)
        df = reduce_categories(df, col)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)

    return df

def align_columns(df, model):
    """Align input data columns to match the model's expected features."""
    model_features = model.feature_names_in_

    # Add missing columns with default value 0
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match the model's training order
    df = df[model_features]

    return df


def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest):
    """Make a prediction using the trained model."""
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
