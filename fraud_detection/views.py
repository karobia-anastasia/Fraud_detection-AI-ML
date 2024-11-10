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
from .data_cleaning import clean_data, split_data, train_and_evaluate_random_forest, save_model_and_scaler, load_model


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

# Handle file upload and either train or predict
def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        form = UploadFileForm(request.POST, request.FILES)
        
        if form.is_valid():
            file = form.cleaned_data['file']
            description = form.cleaned_data.get('description', '')  # Optional description
            
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            
            logger.info(f"File uploaded: {filename}, path: {file_path}")
            logger.info(f"File description: {description}")

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

            try:
                # Try to load the existing model
                model = load_model()  # Only load the model, no scaler needed
                logger.info("Model found, making predictions...")

                # Align the input features with the model's expected features
                cleaned_df = align_columns(cleaned_df, model)

                # Make predictions
                predictions = model.predict(cleaned_df)

                # If you need to evaluate on a separate test set, you would also drop 'isFraud' here
                if 'Fraud' in df.columns:
                    accuracy = model.score(cleaned_df, df['Fraud'])  # Check accuracy, if 'Fraud' is in the test set
                    logger.info(f"Model found. Accuracy: {accuracy * 100:.2f}%")
                else:
                    accuracy = "No accuracy calculation available (no test labels)."

                # After processing, redirect to the transactions page with the predictions
                return redirect('transactions')  # Use the URL name for the transaction list page
                
            except FileNotFoundError:
                # If model doesn't exist, train the model
                model = train_and_evaluate_random_forest(cleaned_df)  # Train the model here
                save_model_and_scaler(model, None)  # Save the trained model (no scaler in this case)

                # After training, redirect to the transactions page
                return redirect('transactions')  # Redirect to transactions page after training

            except EOFError as e:
                # Handle case where model file is corrupted
                logger.error(f"Error loading model: {e}")
                return render(request, 'error_page.html', {'error': f"Model file is corrupted. Please re-train the model."})

    else:
        form = UploadFileForm()

    return render(request, 'upload_file.html', {'form': form})

def input_transaction(request):
    if request.method == 'POST':
        form = TransactionForm(request.POST)
        
        if form.is_valid():
            # Extract data from the form
            step = form.cleaned_data['step']
            transaction_type = form.cleaned_data['type']
            amount = form.cleaned_data['amount']
            name_orig = form.cleaned_data['nameOrig']
            old_balance_orig = form.cleaned_data['oldbalanceOrg']
            new_balance_orig = form.cleaned_data['newbalanceOrig']
            name_dest = form.cleaned_data['nameDest']
            old_balance_dest = form.cleaned_data['oldbalanceDest']
            new_balance_dest = form.cleaned_data['newbalanceDest']
            
            # Save the transaction to the database
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

            # Log the transaction to make sure it's saved
            logger.info(f"Transaction saved: {transaction}")
            
            # Optionally: Redirect after saving the transaction (to avoid re-submitting the form)
            return redirect('transactions')  # Redirect to a page that lists the transactions, e.g., 'transactions'
        else:
            # If the form is invalid, log the errors
            logger.error(f"Form is invalid: {form.errors}")
            return render(request, 'input_transaction.html', {'form': form, 'error': 'Form is invalid. Please check the fields.'})
    
    else:
        form = TransactionForm()

    # Render the form on GET request
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


def prediction_reports(request):
    transactions = Transaction.objects.all()
    transaction_df = pd.DataFrame(list(transactions.values()))
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    X = transaction_df[features]
    y = transaction_df['isFraud']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    # Train a RandomForestClassifier model with balanced class weights to handle class imbalance
    model = RandomForestClassifier(class_weight='balanced')
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
