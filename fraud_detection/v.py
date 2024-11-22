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
                model = train_and_evaluate_model(cleaned_df)  # Train the model here
                save_model(model, None)  # Save the trained model (no scaler in this case)

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

    # Using .iloc to avoid the FutureWarning
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Iterate over the transactions and update the prediction fields
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    # Generate the classification report and confusion matrix
    report = classification_report(y, predictions, output_dict=True)
    cm = confusion_matrix(y, predictions)
    confusion_image = plot_confusion_matrix(cm)

    # Plot prediction counts
    prediction_counts = pd.Series(predictions).value_counts()
    prediction_chart = plot_prediction_counts(prediction_counts)

    prediction_data = {
        'labels': ['Non-Fraud', 'Fraud'],
        'data': [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]
    }

    # Render the dashboard view with the results
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


    # Function to compare the models and display the results
def compare_and_plot_models(df):
    # Clean the data
    df_cleaned = clean_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_cleaned)

    # Train and evaluate XGBoost model (Base model)
    xgb_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='XGBoost')

    # Train and evaluate RandomForest model (Proposed model)
    rf_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='RandomForest')

    # Train and evaluate LogisticRegression model (Proposed model)
    lr_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='LogisticRegression')

    # Calculate percentage improvement of RandomForest over XGBoost
    if xgb_accuracy > 0:
        rf_improvement_percentage = ((rf_accuracy - xgb_accuracy) / xgb_accuracy) * 100
    else:
        rf_improvement_percentage = float('inf')  # Handle case where base model accuracy is zero

    # Calculate percentage improvement of LogisticRegression over XGBoost
    if xgb_accuracy > 0:
        lr_improvement_percentage = ((lr_accuracy - xgb_accuracy) / xgb_accuracy) * 100
    else:
        lr_improvement_percentage = float('inf')  # Handle case where base model accuracy is zero

    print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
    print(f"RandomForest Accuracy: {rf_accuracy * 100:.2f}% (Improvement: {rf_improvement_percentage:.2f}%)")
    print(f"LogisticRegression Accuracy: {lr_accuracy * 100:.2f}% (Improvement: {lr_improvement_percentage:.2f}%)")

    # Plot the performance comparison after the accuracy values are computed
    plot_performance(xgb_accuracy, rf_accuracy, lr_accuracy)



# Plotting function to compare model performances
def plot_performance(xgb_accuracy, rf_accuracy, lr_accuracy):
    models = ['XGBoost', 'RandomForest', 'LogisticRegression']
    accuracies = [xgb_accuracy, rf_accuracy, lr_accuracy]

    plt.bar(models, accuracies, color=['blue', 'green', '#30638e'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)  # Accuracy range from 0 to 1

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v * 100:.2f}%', ha='center')

    plt.show()



from django.shortcuts import render
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from django.conf import settings
from .models import Transaction  # Assuming the model is called Transaction
import os
import joblib
import base64
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from django.shortcuts import render
from .models import Transaction
from django.conf import settings

# Function to render the prediction report
def prediction_reports(request):
    # Fetch transactions and prepare DataFrame
    transactions = Transaction.objects.all()
    transaction_df = pd.DataFrame(list(transactions.values()))
    
    # Features and target for model
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X = transaction_df[features]
    y = transaction_df['isFraud']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    # Train a RandomForest model with balanced class weights
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # Save the trained model to disk
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Use the model to make predictions on all transactions
    predictions = model.predict(X)

    # Add predictions to the DataFrame
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Iterate over the transactions and save prediction results
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    # Generate classification report and confusion matrix
    report = classification_report(y, predictions, output_dict=True)
    cm = confusion_matrix(y, predictions)
    confusion_image = plot_confusion_matrix(cm)

    # Plot prediction counts
    prediction_counts = pd.Series(predictions).value_counts()
    prediction_chart = plot_prediction_counts(prediction_counts)

    # Prepare data for rendering
    prediction_data = {
        'labels': ['Non-Fraud', 'Fraud'],
        'data': [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]
    }

    # Generate performance bar chart
    performance_bar_chart = plot_performance_bar_chart(report)

    # Prepare data for model comparison chart
    comparison_chart = create_comparison_chart(report)

    # Render results to the dashboard view
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
        'prediction_chart': prediction_chart,
        'performance_bar_chart': performance_bar_chart,
        'comparison_chart': comparison_chart
    })

# Function to plot confusion matrix with blue shades
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16, color='#6a4c9c')  # Purple color for title
    plt.ylabel('Actual', fontsize=12, color='#6a4c9c')
    plt.xlabel('Predicted', fontsize=12, color='#6a4c9c')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img_to_base64(img)

# Function to plot prediction counts (Fraud vs Non-Fraud) with blue tones
def plot_prediction_counts(prediction_counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette='Blues')
    ax.set_title('Prediction Counts: Fraud vs Non-Fraud', fontsize=16, color='#6a4c9c')  # Purple title
    ax.set_xlabel('Prediction', fontsize=12, color='#6a4c9c')
    ax.set_ylabel('Count', fontsize=12, color='#6a4c9c')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)

# Function to plot the performance of the RandomForest model in blue theme
def plot_performance_bar_chart(report):
    # Metrics to be displayed dynamically from the classification report
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        report['accuracy'] * 100,  # Accuracy in percentage
        report['weighted avg']['precision'] * 100,  # Precision in percentage
        report['weighted avg']['recall'] * 100,  # Recall in percentage
        report['weighted avg']['f1-score'] * 100  # F1-Score in percentage
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics, values, color='blue')  # Blue bars

    # Add percentage labels on top of each bar
    for i, value in enumerate(values):
        ax.text(i, value + 0.5, f'{value:.2f}%', ha='center', fontsize=12)

    ax.set_ylabel('Percentage (%)', fontsize=12, color='#6a4c9c')  # Purple color for labels
    ax.set_title('Model Performance', fontsize=16, color='#6a4c9c')  # Purple title
    ax.set_ylim(0, 100)  # Ensures the y-axis ranges from 0 to 100%

    # Convert to base64 image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)

# Helper function to convert image to base64
def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()

# Helper function to create a model comparison chart (Accuracy, Precision, Recall, F1-Score in percentage)
def create_comparison_chart(report):
    # Example comparison data (for now, this could be dynamically generated from other models if needed)
    models = ['RandomForest']  # You can extend this list with other models if you compare them
    accuracy = [report['accuracy'] * 100]  # Accuracy as percentage
    precision = [report['weighted avg']['precision'] * 100]  # Precision as percentage
    recall = [report['weighted avg']['recall'] * 100]     # Recall as percentage
    f1_score = [report['weighted avg']['f1-score'] * 100]   # F1-Score as percentage

    # Plot performance comparison for the models
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define the x-axis positions and width for each model's bar
    index = range(len(models))
    bar_width = 0.2  # Width of the bars

    # Plot each metric as a separate set of bars
    ax.bar([i - bar_width for i in index], accuracy, bar_width, label='Accuracy', color='blue')
    ax.bar(index, precision, bar_width, label='Precision', color='green')
    ax.bar([i + bar_width for i in index], recall, bar_width, label='Recall', color='orange')
    ax.bar([i + 2*bar_width for i in index], f1_score, bar_width, label='F1-Score', color='red')

    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(index)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)  # Set y-axis to show percentage range

    # Adding a legend
    ax.legend()

    # Convert the plot to base64 image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img_to_base64(img)
