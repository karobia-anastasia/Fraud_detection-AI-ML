import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from django.conf import settings
from fraud_detection.data_cleaning import clean_data, save_model, split_data, align_columns

# Helper function for cleaning data and training the model
def clean_and_train_model(file_path):
    # Load data
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    # Clean the data
    cleaned_df = clean_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(cleaned_df)
    
    # Train the model (using Random Forest here)
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    save_model(model, model_path)
    
    return model

# Function to make predictions with the trained model
def make_predictions(model, X):
    return model.predict(X)

# Generate performance report
def generate_report(transactions, predictions):
    y = pd.Series([t.isFraud for t in transactions])
    report = classification_report(y, predictions, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
    }
