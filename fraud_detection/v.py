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

import logging
import base64
from io import BytesIO
from django.shortcuts import render
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .models import Transaction
from django.conf import settings
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)

# Function to convert image to base64 encoding
def img_to_base64(img):
    return base64.b64encode(img.getvalue()).decode()


# Function to plot the confusion matrix
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

# Function to plot prediction counts (Fraud vs Non-Fraud)
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

def prediction_reports(request):
    # Retrieve all transactions
    transactions = Transaction.objects.all()
    transaction_df = pd.DataFrame(list(transactions.values()))

    # Select the features you want to use for prediction
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # Ensure that all feature columns are numeric
    for col in features:
        transaction_df[col] = pd.to_numeric(transaction_df[col], errors='coerce')  # Convert to numeric, coerce errors to NaN

    # Fill any NaN values that resulted from coercion
    transaction_df[features] = transaction_df[features].fillna(transaction_df[features].median())

    X = transaction_df[features]  # Features for prediction
    y = transaction_df['isFraud']  # Labels

    # Load the pre-trained model
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError('Model file not found. Please train the model first.')

    # Make predictions using the model
    predictions = model.predict(X)

    # Add predictions and prediction labels to the DataFrame
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Save predictions to the database
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    # Prediction counts: Number of Fraud vs Non-Fraud predictions
    prediction_counts = {
        'Non-Fraud': sum([t.prediction == 0 for t in transactions]),
        'Fraud': sum([t.prediction == 1 for t in transactions]),
    }

    # Generate classification report
    report = classification_report(y, predictions, output_dict=True)

    # Confusion Matrix
    cm = confusion_matrix(y, predictions)
    confusion_image = plot_confusion_matrix(cm)

    # Prediction counts visualization
    prediction_counts_series = pd.Series(prediction_counts).sort_index()
    prediction_chart = plot_prediction_counts(prediction_counts_series)

    # Comparison bar chart: Random Forest, XGBoost, Logistic Regression
    model_performance = compare_models(X, y)

    # Debugging - log the base64 data to check for empty graphs
    logger.debug(f"Performance Bar Chart Base64: {model_performance['accuracy_comparison_img'][:100]}")
    logger.debug(f"Confusion Matrix Base64: {confusion_image[:100]}")
    logger.debug(f"Prediction Counts Chart Base64: {prediction_chart[:100]}")

    # Format the performance metrics as percentages
    best_model_name = model_performance['best_model_name']
    performance_data = model_performance['model_performance'][best_model_name]
    formatted_report = {
        'accuracy': f"{performance_data['accuracy']:.2f}",  # Already converted to percentage
        'precision': f"{performance_data['precision'] * 100:.2f}",
        'recall': f"{performance_data['recall'] * 100:.2f}",
        'f1_score': f"{performance_data['f1_score'] * 100:.2f}",
    }

    # Create the performance bar graph for the best model
    performance_bar_graph = plot_performance_bar_graph(performance_data)

    # Render the template with formatted data
    return render(request, 'dashboard_view.html', {
        'best_model_name': best_model_name,
        'report': formatted_report,
        'performance_bar_graph': performance_bar_graph,  # Base64 image of the performance bar graph
        'comparison_chart': model_performance['accuracy_comparison_img'],  # Base64 image for accuracy comparison
        'xgb_improvement': model_performance['xgb_improvement'],  # XGBoost improvement %
        'logreg_improvement': model_performance['logreg_improvement'],  # Logistic Regression improvement %
        'confusion_image': confusion_image,  # Base64 image of confusion matrix
        'prediction_chart': prediction_chart  # Base64 image of prediction counts
    })

def compare_models(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LogisticRegression': LogisticRegression(solver='liblinear', class_weight='balanced')
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ''
    model_performance = {}
    accuracy_comparison = {}

    # Training and evaluating each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_percentage = accuracy * 100  # Convert to percentage

        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

        # Store the model performance, including accuracy as percentage
        model_performance[model_name] = {
            'accuracy': accuracy_percentage,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        accuracy_comparison[model_name] = accuracy_percentage  # Store just accuracy for the bar chart

        # Update best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = model

    # Logging model performance for debugging
    logger.debug(f"Model Performance (Accuracy %): {model_performance}")

    # Bar chart comparing model accuracy
    accuracy_comparison_img = plot_comparison_bar_chart(accuracy_comparison)

    # Calculate improvements in accuracy
    xgb_improvement = model_performance['RandomForest']['accuracy'] - model_performance['XGBoost']['accuracy']
    logreg_improvement = model_performance['RandomForest']['accuracy'] - model_performance['LogisticRegression']['accuracy']

    return {
        'model_performance': model_performance,
        'best_model_name': best_model_name,
        'accuracy_comparison_img': accuracy_comparison_img,
        'xgb_improvement': xgb_improvement,
        'logreg_improvement': logreg_improvement
    }
def plot_comparison_bar_chart(accuracy_comparison):
    # Create a bar chart for accuracy comparison
    fig, ax = plt.subplots()
    ax.bar(accuracy_comparison.keys(), accuracy_comparison.values())
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')

    # Convert plot to base64
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str
# Function to plot the model performance as a bar graph
def plot_performance_bar_graph(performance_data):
    # Extracting the metrics for the best model
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        performance_data['accuracy'],
        performance_data['precision'],
        performance_data['recall'],
        performance_data['f1_score']
    ]
    
    # Create the bar graph for model performance
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(metrics, values, color=['#4CAF50', '#FFC107', '#2196F3', '#FF5722'])

    ax.set_title('Model Performance Metrics')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)  # Y-axis range from 0 to 100%

    # Adding values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center')

    # Save the plot as a base64 image
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Close the plot to avoid display in the notebook

    return img_str
