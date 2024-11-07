import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

def train_models(data):
    # Prepare features and labels
    X = data.drop(columns=['isFraud'])
    y = data['isFraud']

    # Split the dataset into training and testing sets (70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, f'model/{model_name.lower().replace(" ", "_")}_model.pkl')

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        predictions = model.predict(X_test)

        results[model_name] = {
            'accuracy': model.score(X_test, y_test),
            'cross_val_mean': cv_scores.mean(),
            'classification_report': classification_report(y_test, predictions)
        }

    return results
