import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(test_data):
    # Prepare features and labels
    X_test = test_data.drop(columns=['isFraud'])
    y_test = test_data['isFraud']

    models = {
        'Random Forest': joblib.load('model/random_forest_model.pkl'),
        'XGBoost': joblib.load('model/xgboost_model.pkl')
    }

    evaluation_results = {}

    for model_name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Generate evaluation metrics
        report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        evaluation_results[model_name] = {
            'report': report,
            'confusion_matrix': cm,
            'auc': auc
        }

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'model/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()

    return evaluation_results
