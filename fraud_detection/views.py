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

logger = logging.getLogger(__name__)

@login_required
def upload_data(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            if file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
                df.columns = df.columns.str.strip()

                for index, row in df.iterrows():
                    transaction, created = Transaction.objects.get_or_create(
                        step=row['step'],
                        amount=row['amount'],
                        type=row['type'],
                        defaults={
                            'isFraud': bool(row['isFraud']),
                            'isFlaggedFraud': bool(row['isFlaggedFraud']),
                            'nameOrig': row['nameOrig'],
                            'oldbalanceOrg': row['oldbalanceOrg'],
                            'newbalanceOrig': row['newbalanceOrig'],
                            'nameDest': row['nameDest'],
                            'oldbalanceDest': row['oldbalanceDest'],
                            'newbalanceDest': row['newbalanceDest'],
                        }
                    )

                    if not created:
                        transaction.step = row['step']
                        transaction.type = row['type']
                        transaction.isFraud = bool(row['isFraud'])
                        transaction.isFlaggedFraud = bool(row['isFlaggedFraud'])
                        transaction.amount = row['amount']
                        transaction.nameOrig = row['nameOrig']
                        transaction.oldbalanceOrg = row['oldbalanceOrg']
                        transaction.newbalanceOrig = row['newbalanceOrig']
                        transaction.nameDest = row['nameDest']
                        transaction.oldbalanceDest = row['oldbalanceDest']
                        transaction.newbalanceDest = row['newbalanceDest']
                        transaction.save()

                return redirect('transactions')

    else:
        form = UploadFileForm()

    return render(request, 'upload_file.html', {'form': form})

@login_required
def transaction_list(request):
    transactions = Transaction.objects.all()
    paginator = Paginator(transactions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'transaction_list.html', {'page_obj': page_obj})

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)
        
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            return render(request, 'upload_file.html', {'error': "Unsupported file type. Please upload a .csv or .xlsx file."})

        cleaned_df = clean_data(df)
        X_train, X_test, y_train, y_test = split_data(cleaned_df)

        # Train the model using the training data
        model = train_model(X_train, y_train, X_test, y_test)

        # Make predictions using the trained model
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        evaluate_model(y_test, y_pred)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        return render(request, 'transaction_list.html', {
            'accuracy': accuracy,
            'predictions': y_pred
        })
    
    return render(request, 'upload_file.html')

@login_required
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

    paginator = Paginator(transactions, 20)
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

@login_required
def prediction_reports(request):
    transactions = Transaction.objects.all()
    transaction_df = pd.DataFrame(list(transactions.values()))
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    X = transaction_df[features]
    y = transaction_df['isFraud']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    # Train a RandomForestClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the trained model to disk
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Use the model to make predictions on all transactions
    predictions = model.predict(X)

    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    report = classification_report(y, predictions, output_dict=True)
    cm = confusion_matrix(y, predictions)
    confusion_image = plot_confusion_matrix(cm)

    prediction_counts = pd.Series(predictions).value_counts()
    prediction_chart = plot_prediction_counts(prediction_counts)

    prediction_data = {
        'labels': ['Non-Fraud', 'Fraud'],
        'data': [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]
    }

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



