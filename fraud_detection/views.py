from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import UploadFileForm, TransactionForm
from .models import Transaction
from fraud_detection.data_cleaning import clean_data, save_model, split_data, load_model, align_columns, train_and_evaluate_model
import pandas as pd
import joblib
import base64
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Setup logging
logger = logging.getLogger(__name__)

def transaction_list(request):
    transactions = Transaction.objects.all()
    paginator = Paginator(transactions, 10)  # 10 per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'transaction_list.html', {'page_obj': page_obj})

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
                model = load_model()  # Only load the model, no scaler needed
                logger.info("Model found, making predictions...")

                # Align the input features with the model's expected features
                cleaned_df = align_columns(cleaned_df, model)

                predictions = model.predict(cleaned_df)

                if 'Fraud' in df.columns:
                    accuracy = model.score(cleaned_df, df['Fraud'])
                    logger.info(f"Model found. Accuracy: {accuracy * 100:.2f}%")
                else:
                    accuracy = "No accuracy calculation available (no test labels)."

                return redirect('transactions')

            except FileNotFoundError:
                model = train_and_evaluate_model(cleaned_df)  # Train the model here
                save_model(model, None)
                return redirect('transactions')

            except EOFError as e:
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
            transaction = Transaction.objects.create(
                step=form.cleaned_data['step'],
                type=form.cleaned_data['type'],
                amount=form.cleaned_data['amount'],
                nameOrig=form.cleaned_data['nameOrig'],
                oldbalanceOrg=form.cleaned_data['oldbalanceOrg'],
                newbalanceOrig=form.cleaned_data['newbalanceOrig'],
                nameDest=form.cleaned_data['nameDest'],
                oldbalanceDest=form.cleaned_data['oldbalanceDest'],
                newbalanceDest=form.cleaned_data['newbalanceDest']
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
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError('Model file not found. Please train the model first.')

    predictions = model.predict(X)

    transaction_df = pd.DataFrame(list(transactions.values()))
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

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
