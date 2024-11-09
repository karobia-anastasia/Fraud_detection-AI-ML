import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from django.conf import settings
from django.shortcuts import render
from .forms import TransactionForm  # Import your form class here

# Function to clean the data
def clean_data(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        df[col].fillna(df[col].median(), inplace=True)  

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip() 
        df[col].fillna(df[col].mode()[0], inplace=True)

    boolean_cols = ['isFraud','isFlaggedFraud']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

# Split the data into training and testing sets
def split_data(df):
    if 'isFraud' not in df.columns:
        raise KeyError("'isFraud' column not found in the dataset")

    X = df.drop('isFraud', axis=1)  # Features
    y = df['isFraud'].astype(int)   # Target (convert to binary if needed)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")
    return model

# Train the Random Forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    return model

# Function to save the model and scaler
def save_model_and_scaler(model, scaler):
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'models', 'scaler.pkl')

    joblib.dump(model, model_path)  # Save the trained model
    joblib.dump(scaler, scaler_path)  # Save the scaler
    print(f"Model and scaler saved to {model_path} and {scaler_path}")

# Function to load the model and scaler for prediction
def load_model_and_scaler():
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'models', 'scaler.pkl')

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        raise FileNotFoundError("Model or scaler not found. Please ensure both are saved and available.")

# View to handle prediction request
def detect_fraud(request):
    if request.method == 'POST':
        form = TransactionForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from form
            transaction_data = {
                'step': form.cleaned_data['step'],
                'amount': form.cleaned_data['amount'],
                'oldbalanceOrg': form.cleaned_data['oldbalanceOrg'],
                'newbalanceOrig': form.cleaned_data['newbalanceOrig'],
                'oldbalanceDest': form.cleaned_data['oldbalanceDest'],
                'newbalanceDest': form.cleaned_data['newbalanceDest'],
            }

            # Convert form data into DataFrame
            X_new = pd.DataFrame([transaction_data])

            try:
                # Load the pre-trained model and scaler
                model, scaler = load_model_and_scaler()

                # Scale the input data using the pre-trained scaler
                X_new_scaled = scaler.transform(X_new)

                # Make the prediction
                prediction = model.predict(X_new_scaled)

                # Determine the result
                result = 'Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent'

            except FileNotFoundError as e:
                result = f"Error: {str(e)}"
            except Exception as e:
                result = f"An error occurred during prediction: {str(e)}"

            # Return result to the template
            return render(request, 'detect_fraud.html', {'form': form, 'result': result})

    else:
        form = TransactionForm()

    return render(request, '.html', {'form': form})
