import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Train and evaluate Random Forest model
def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    
    # Return the trained model
    return model


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

# Save the trained model and scaler
def save_model_and_scaler(model):
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')

    # Save the model to a file
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Load the model for prediction
def load_model():
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')

    # Check if the model exists
    if os.path.exists(model_path):
        try:
            # Try to load model
            model = joblib.load(model_path)
            return model
        except EOFError:
            # If EOFError occurs (corrupted model file), raise error
            raise EOFError("Model file appears to be corrupted. Please re-train the model.")
    else:
        raise FileNotFoundError("Model not found. Please ensure the model is saved and available.")

# Preprocess the input data for prediction
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
def predict(step, transaction_type, amount, name_orig, old_balance_orig, new_balance_orig, name_dest, old_balance_dest, new_balance_dest):
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
    model = load_model()

    # Preprocess the data (no scaler needed)
    processed_data = preprocess_for_prediction(transaction_data)

    # Make the prediction
    prediction = model.predict(processed_data)

    # Return the prediction result (fraudulent or not)
    return "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"
