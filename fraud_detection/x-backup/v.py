from django.contrib.auth.decorators import login_required
import json
import base64
import os
import traceback
import joblib
import pandas as pd
from io import BytesIO
from django.shortcuts import render, redirect
from fraud_detection.data_cleaning import *
from fraud_detection.forms import UploadFileForm
from django.http import JsonResponse
from sklearn.metrics import classification_report, confusion_matrix
from .forms import TransactionForm, UploadFileForm
from django.core.files.storage import FileSystemStorage
from .models import Transaction
from django.core.paginator import Paginator
from django.conf import settings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from .data_cleaning import clean_data, split_data, load_model


import os
import joblib
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings

from sklearn.metrics import classification_report, accuracy_score


logger = logging.getLogger(__name__)
def transaction_list(request):
    transactions = Transaction.objects.all()  
    
    # Paginate the transactions
    paginator = Paginator(transactions, 10)  # 10 per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Pass the page_obj to the template
    return render(request, 'transaction_list.html', {'page_obj': page_obj})

# Setup logging
logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Function to handle file upload and train/predict workflow
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        form = UploadFileForm(request.POST, request.FILES)
        
        if form.is_valid():
            file = form.cleaned_data['file']
            description = form.cleaned_data.get('description', '')  # Optional description
            
            # Save file
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            
            logger.info(f"File uploaded: {filename} | Path: {file_path}")
            if description:
                logger.info(f"File description: {description}")

            # Step 1: Read the File
            try:
                if file.name.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif file.name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                return render(request, 'upload_file.html', {'error': str(e)})

            # Step 2: Clean Data
            try:
                cleaned_df = clean_data(df)
                logger.info("Data cleaned successfully.")
            except Exception as e:
                logger.error(f"Error cleaning data: {e}")
                return render(request, 'upload_file.html', {'error': f"Data cleaning failed: {e}"})

            # Step 3: Check for Existing Model
            try:
                model = load_model()
                logger.info("Existing model loaded successfully. Proceeding with predictions...")

                # Align columns to model input
                cleaned_df = align_columns(cleaned_df, model)
                
                # Predict
                predictions = model.predict(cleaned_df)
                logger.info("Predictions completed successfully.")

                # Calculate accuracy if labels exist
                if 'Fraud' in cleaned_df.columns:
                    accuracy = model.score(cleaned_df.drop(columns=['Fraud']), cleaned_df['Fraud'])
                    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
                else:
                    accuracy = "No labels found. Accuracy calculation skipped."

                # Save predictions if needed
                cleaned_df['Predictions'] = predictions
                output_file = file_path.replace('.csv', '_predictions.csv')
                cleaned_df.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to: {output_file}")

                return redirect('transactions')  # Adjust based on your app's flow

            except (FileNotFoundError, EOFError) as e:
                logger.warning("No model found or model file corrupted. Training a new model...")
                
                # Step 4: Train New Model
                try:
                    X_train, X_test, y_train, y_test = split_data(cleaned_df)
                    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
                    save_model(model_results['xgb_model'], None)
                    
                    logger.info("New model trained and saved successfully.")
                    return redirect('transactions')
                except Exception as e:
                    logger.error(f"Model training failed: {e}")
                    return render(request, 'error_page.html', {'error': f"Training failed: {e}"})

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return render(request, 'error_page.html', {'error': f"An unexpected error occurred: {e}"})
    
    else:
        form = UploadFileForm()

    return render(request, 'upload_file.html', {'form': form})


def input_transaction(request):
    if request.method == 'POST':
        form = TransactionForm(request.POST)
        
        if form.is_valid():
            step = form.cleaned_data['step']
            transaction_type = form.cleaned_data['type']
            amount = form.cleaned_data['amount']
            name_orig = form.cleaned_data['nameOrig']
            old_balance_orig = form.cleaned_data['oldbalanceOrg']
            new_balance_orig = form.cleaned_data['newbalanceOrig']
            name_dest = form.cleaned_data['nameDest']
            old_balance_dest = form.cleaned_data['oldbalanceDest']
            new_balance_dest = form.cleaned_data['newbalanceDest']            
            transaction = Transaction.objects.create(
                step=step,
                type=transaction_type,
                amount=amount,
                nameOrig=name_orig,
                oldbalanceOrg=old_balance_orig,
                newbalanceOrig=new_balance_orig,
                nameDest=name_dest,
                oldbalanceDest=old_balance_dest,
                newbalanceDest=new_balance_dest
            )

            logger.info(f"Transaction saved: {transaction}")
            
            return redirect('transactions') 
        else:
            logger.error(f"Form is invalid: {form.errors}")
            return render(request, 'input_transaction.html', {'form': form, 'error': 'Form is invalid. Please check the fields.'})
    
    else:
        form = TransactionForm()

    return render(request, 'input_transaction.html', {'form': form})

def prediction_results(request):
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

    transaction_df = pd.DataFrame(list(transactions.values()))
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Update each transaction's prediction in the database
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    prediction_counts = {
        'Non-Fraud': sum([t.prediction == 0 for t in transactions]),
        'Fraud': sum([t.prediction == 1 for t in transactions]),
    }

    y = transaction_df['isFraud']
    report = classification_report(y, predictions, output_dict=True)

    paginator = Paginator(transactions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'data_analysis.html', {
        'page_obj': page_obj,
        'report': {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
        },
        'prediction_data': prediction_counts
    })

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    xgboost_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(
        eval_metric='logloss', 
        **xgboost_params
    )
    xgb_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)

    rf_report = classification_report(y_test, rf_predictions, output_dict=True)
    xgb_report = classification_report(y_test, xgb_predictions, output_dict=True)

    try:
        rf_precision = rf_report['True']['precision']
        rf_recall = rf_report['True']['recall']
        rf_f1_score = rf_report['True']['f1-score']

        xgb_precision = xgb_report['True']['precision']
        xgb_recall = xgb_report['True']['recall']
        xgb_f1_score = xgb_report['True']['f1-score']

    except KeyError as e:
        print(f"Error: Class label 'True' not found in the classification report. Available labels: {list(rf_report.keys())}")
        raise e

    rf_accuracy = accuracy_score(y_test, rf_predictions)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    if rf_accuracy == 0:  
        performance_improvement = None
    else:
        performance_improvement = ((xgb_accuracy - rf_accuracy) / rf_accuracy) * 100

    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'performance_improvement': performance_improvement,
        'rf_accuracy': rf_accuracy,
        'xgb_accuracy': xgb_accuracy,
        'rf_precision': rf_precision,
        'rf_recall': rf_recall,
        'rf_f1_score': rf_f1_score,
        'xgb_precision': xgb_precision,
        'xgb_recall': xgb_recall,
        'xgb_f1_score': xgb_f1_score,
        'xgboost_params': xgboost_params
    }


def prediction_reports(request):
    try:
        transactions = Transaction.objects.all()
        transaction_df = pd.DataFrame(list(transactions.values()))

        transaction_df['amount'] = pd.to_numeric(transaction_df['amount'], errors='coerce')
        transaction_df['oldbalanceOrg'] = pd.to_numeric(transaction_df['oldbalanceOrg'], errors='coerce')
        transaction_df['newbalanceOrig'] = pd.to_numeric(transaction_df['newbalanceOrig'], errors='coerce')
        transaction_df['oldbalanceDest'] = pd.to_numeric(transaction_df['oldbalanceDest'], errors='coerce')
        transaction_df['newbalanceDest'] = pd.to_numeric(transaction_df['newbalanceDest'], errors='coerce')
        required_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
        transaction_df = transaction_df.dropna(subset=required_columns)
        if transaction_df.empty:
            return HttpResponse("No valid data available for training and prediction.")
        features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        X = transaction_df[features]
        y = transaction_df['isFraud']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        evaluation_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        rf_model = evaluation_results['rf_model']
        xgb_model = evaluation_results['xgb_model']
        rf_accuracy = evaluation_results['rf_accuracy']
        xgb_accuracy = evaluation_results['xgb_accuracy']
        performance_improvement = evaluation_results['performance_improvement']
        rf_precision = evaluation_results['rf_precision']
        rf_recall = evaluation_results['rf_recall']
        rf_f1_score = evaluation_results['rf_f1_score']
        xgb_precision = evaluation_results['xgb_precision']
        xgb_recall = evaluation_results['xgb_recall']
        xgb_f1_score = evaluation_results['xgb_f1_score']
        xgboost_params = evaluation_results['xgboost_params']

        cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))
        cm_xgb = confusion_matrix(y_test, xgb_model.predict(X_test))
        confusion_image_rf = plot_confusion_matrix(cm_rf)
        confusion_image_xgb = plot_confusion_matrix(cm_xgb)

        # Make predictions with both models
        rf_predictions = rf_model.predict(X_test)
        xgb_predictions = xgb_model.predict(X_test)

        # Calculate prediction counts for both models (Fraud and Non-Fraud)
        prediction_counts_rf = pd.Series(rf_predictions).value_counts()
        prediction_counts_xgb = pd.Series(xgb_predictions).value_counts()

        # Plot the prediction counts as graphs
        prediction_chart_rf = plot_prediction_counts(prediction_counts_rf)
        prediction_chart_xgb = plot_prediction_counts(prediction_counts_xgb)

        # Prepare prediction data for both Random Forest and XGBoost
        prediction_data = {
            'labels': ['Non-Fraud', 'Fraud'],
            'data': [
                prediction_counts_rf.get(0, 0),  # Non-Fraud (RF)
                prediction_counts_rf.get(1, 0),  # Fraud (RF)
                prediction_counts_xgb.get(0, 0), # Non-Fraud (XGB)
                prediction_counts_xgb.get(1, 0)  # Fraud (XGB)
            ]
        }

        # Prepare context for template
        context = {
            'rf_accuracy': rf_accuracy,
            'xgb_accuracy': xgb_accuracy,
            'performance_improvement_xgb': performance_improvement,
            'rf_precision': rf_precision,
            'rf_recall': rf_recall,
            'rf_f1_score': rf_f1_score,
            'xgb_precision': xgb_precision,
            'xgb_recall': xgb_recall,
            'xgb_f1_score': xgb_f1_score,
            'cm_chart_rf': confusion_image_rf,
            'cm_chart_xgb': confusion_image_xgb,
            'xgboost_algorithm': xgboost_params,
            'confusion_image_rf': confusion_image_rf,
            'confusion_image_xgb': confusion_image_xgb,
            'performance_comparison_chart': plot_comparison_graph(rf_accuracy, xgb_accuracy, performance_improvement),
            'prediction_chart_rf': prediction_chart_rf,
            'prediction_chart_xgb': prediction_chart_xgb,
            'prediction_data': prediction_data,
        }

        return render(request, 'dashboard_view.html', context)

    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error in prediction_reports: {e}")
        return HttpResponse(f"An error occurred while generating the reports: {e}")


import matplotlib.pyplot as plt
import base64
from io import BytesIO

def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_comparison_graph(base_accuracy, xgb_accuracy, improvement):
    plt.figure(figsize=(8, 6))
    plt.bar(['RandomForest', 'XGBoost'], [base_accuracy, xgb_accuracy], color=['blue', 'orange'])
    plt.title(f'Accuracy Comparison (Improvement: {improvement:.2f}%)')
    plt.ylabel('Accuracy')
    
    img = BytesIO()
    plt.savefig(img, format='png') 
    img.seek(0)  
    plt.close() 
    return img_to_base64(img)
    

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['Non-Fraud', 'Fraud'])
    ax.set_yticklabels(['Non-Fraud', 'Fraud'])
    plt.colorbar(ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues))
    plt.show()

    return fig

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)

def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()

def plot_prediction_counts(prediction_counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette='Blues')
    ax.set_title('Prediction Counts: Fraud vs Non-Fraud')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Count')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)

def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()