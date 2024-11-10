def upload_and_train_model(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)
        
        # Load the data based on the file extension
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            return render(request, 'upload_file.html', {'error': "Unsupported file type. Please upload a .csv or .xlsx file."})

        # Clean and prepare the data
        cleaned_df = clean_data(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(cleaned_df)

        # Train and evaluate the Random Forest model
        model = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

        # Calculate accuracy score directly
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100  # Calculate accuracy as a percentage

        # Optionally, save the model and scaler if needed
        save_model_and_scaler(model, StandardScaler())  # Example: saving the model and scaler

        # Return the result to the template with accuracy
        return render(request, 'transaction_list.html', {
            'accuracy': accuracy,  # Pass accuracy to the template
            'predictions': y_pred.tolist()  # You can also pass predictions if needed
        })

    return render(request, 'upload_file.html')

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from django.conf import settings

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

# Train and evaluate Logistic Regression model
def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")
    
    # Model Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# Train and evaluate Random Forest model
def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    
    # Model Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
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


