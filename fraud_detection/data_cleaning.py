import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

def clean_data(df):

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        df[col].fillna(df[col].median(), inplace=True)  

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip() 
        df[col].fillna(df[col].mode()[0], inplace=True)

    boolean_cols =  ['isFraud','isFlaggedFraud']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df



def split_data(df):
    # Ensure 'isFraud' column exists
    if 'isFraud'not in df.columns:
        raise KeyError("'isFraud' column not found in the dataset")

    X = df.drop('isFraud', axis=1)  # Features
    y = df['isFraud'].astype(int)   # Target (convert to binary if needed)

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)



def train_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)  # Increase max_iter for larger datasets
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")
    return model


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    return model


def evaluate_model(y_test, y_pred):
    # Evaluate using confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")