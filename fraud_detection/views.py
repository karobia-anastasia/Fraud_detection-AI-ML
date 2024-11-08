import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from fraud_detection.data_cleaning import *
from fraud_detection.forms import UploadFileForm
from fraud_detection.models import Transaction
from django.http import JsonResponse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from django.shortcuts import render, redirect
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage
from .models import Transaction
import pandas as pd
from django.core.paginator import Paginator
import json
from django.conf import settings
import pandas as pd
from django.shortcuts import render, redirect
import joblib
import os
import joblib
import pandas as pd
import os
from django.shortcuts import render
from sklearn.metrics import classification_report, confusion_matrix
from .models import Transaction
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.conf import settings
from .forms import UploadFileForm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report
from django.shortcuts import render
from .models import Transaction
from django.conf import settings
from sklearn.metrics import confusion_matrix
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def upload_data(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle file upload
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)

            # Read the file into a DataFrame
            if file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')

                # Strip any extra spaces in column names
                df.columns = df.columns.str.strip()

                # Process and save each row to the transaction model
                for index, row in df.iterrows():
                    # For rows without 'nameOrig', use None or a default value
    

                    # Check if transaction already exists based on other fields (e.g., step, amount, type)
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

                    # If created is False, the transaction already exists, and you can handle updates if needed.
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

                        # Save the updated transaction object
                        transaction.save()

                return redirect('transactions')  # Redirect to the transaction list page after successful upload

    else:
        form = UploadFileForm()

    return render(request, 'upload_file.html', {'form': form})

def transaction_list(request):
    transactions = Transaction.objects.all()  # Fetch all transactions
    paginator = Paginator(transactions, 10)  # Show 10 transactions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'transaction_list.html', {'page_obj': page_obj})



def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        # Handle file upload
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)
        
        # Load the file into a DataFrame
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # Unsupported file type
            return render(request, 'upload_file.html', {'error': "Unsupported file type. Please upload a .csv or .xlsx file."})

        # Step 1: Clean and preprocess data
        cleaned_df = clean_data(df)
        
        # Step 2: Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(cleaned_df)
        
        # Step 3: Train the model (using Logistic Regression for now)
        model = train_model(X_train, y_train, X_test, y_test)
        
        # Step 4: Make predictions
        y_pred = model.predict(X_test)
        
        # Step 5: Evaluate the model
        evaluate_model(y_test, y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        # Return predictions and accuracy to template

        return render(request, 'transaction_list.html', {
            'accuracy': accuracy,
            'predictions': y_pred
        })
    
    return render(request, 'upload_file.html')




# Set up logging
logger = logging.getLogger(__name__)
from django.core.paginator import Paginator

def prediction_results(request):
    # Load all transactions from the database as model instances (not dictionaries)
    transactions = Transaction.objects.all()

    # Select relevant features (excluding 'isFraud' which is the target)
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X = pd.DataFrame(list(transactions.values()))[features]  # Convert to DataFrame for prediction

    # Load the trained fraud detection model
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError('Model file not found. Please train the model first.')

    # Make predictions for all transactions
    predictions = model.predict(X)

    # Add predictions to the DataFrame for reporting purposes
    transaction_df = pd.DataFrame(list(transactions.values()))  # Create DataFrame
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Update the database with the prediction results (save predictions in the Transaction model)
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]  # Save prediction (0 or 1)
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'  # Save the label
        transaction.save()  # Ensure that the changes are saved in the database

    # Pass transactions directly (use model instances for better access to fields)
    prediction_counts = {
        'Non-Fraud': sum([t.prediction == 0 for t in transactions]),
        'Fraud': sum([t.prediction == 1 for t in transactions]),
    }

    # Prepare classification report
    y = transaction_df['isFraud']
    report = classification_report(y, predictions, output_dict=True)

    # Pagination
    paginator = Paginator(transactions, 20)  # Show 20 transactions per page
    page_number = request.GET.get('page')  # Get the page number from the request
    page_obj = paginator.get_page(page_number)

    # Render the template with the correct data
    return render(request, 'data_analysis.html', {
        'page_obj': page_obj,  # The paginated list of transactions
        'report': {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
        },
        'prediction_data': prediction_counts
    })


def prediction_reports(request):
    # Load all transactions from the database
    transactions = Transaction.objects.all()
    transaction_df = pd.DataFrame(list(transactions.values()))

    # Select relevant features (excluding 'isFraud' which is the target)
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Separate features and target
    X = transaction_df[features]  # Use the correct DataFrame: transaction_df
    y = transaction_df['isFraud']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    # Train the model (RandomForestClassifier)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Define model path and save the trained model
    model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')  # Make sure the filename matches
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the 'models' directory exists
    joblib.dump(model, model_path)  # Save the trained model

    # Make predictions for all transactions (use X for predicting)
    predictions = model.predict(X)

    # Add predictions to the dataframe
    transaction_df['prediction'] = predictions
    transaction_df['prediction_label'] = ['Fraud' if pred == 1 else 'Non-Fraud' for pred in predictions]

    # Update the database with the prediction results
    for idx, transaction in enumerate(transactions):
        transaction.prediction = predictions[idx]
        transaction.prediction_label = 'Fraud' if predictions[idx] == 1 else 'Non-Fraud'
        transaction.save()

    # Prepare classification report
    report = classification_report(y, predictions, output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(y, predictions)
    confusion_image = plot_confusion_matrix(cm)

    # Create bar chart for prediction counts (fraud vs non-fraud)
    prediction_counts = pd.Series(predictions).value_counts()
    prediction_chart = plot_prediction_counts(prediction_counts)

    # Prepare prediction data for the template (bar chart data)
    prediction_data = {
        'labels': ['Non-Fraud', 'Fraud'],
        'data': [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]  # Non-Fraud: 0, Fraud: 1
    }

    # Return results to the template
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


# Function to plot confusion matrix as an image
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    return img_to_base64(img)

# Function to plot prediction counts as a bar chart
def plot_prediction_counts(prediction_counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette='Blues')
    ax.set_title('Prediction Counts: Fraud vs Non-Fraud')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Count')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert to base64 for embedding in HTML
    return img_to_base64(img)

# Helper function to convert image to base64 encoding
def img_to_base64(img):
    # Convert image to base64 encoded string
    return base64.b64encode(img.getvalue()).decode()

# def prediction_results(request):
#     # Load transaction data and proceed as before
#     transactions = Transaction.objects.all()
#     transaction_df = pd.DataFrame(list(transactions.values()))

#     transaction_id_col = transaction_df['transactionID']
#     transaction_df['monthly_charges'] = pd.to_numeric(transaction_df['monthly_charges'], errors='coerce')
#     transaction_df['total_charges'] = pd.to_numeric(transaction_df['total_charges'], errors='coerce')
#     transaction_df.fillna(0, inplace=True)
#     transaction_df = clean_data(transaction_df.drop(columns=['transactionID']))
#     transaction_df['transactionID'] = transaction_id_col

#     X = transaction_df.drop(columns=['churn', 'transactionID'])
#     y = transaction_df['churn']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
    
#     # Check if the confusion matrix is 2x2
#     if cm.shape == (2, 2):
#         tn, fp, fn, tp = cm.ravel()
#     else:
#         # Handle case where there is not enough data for a full confusion matrix
#         tn = fp = fn = tp = 0

#     # Additional metrics: precision, recall, f1_score
#     precision = report['weighted avg']['precision']
#     recall = report['weighted avg']['recall']
#     f1_score = report['weighted avg']['f1-score']

#     # Prediction counts (unchanged from before)
#     prediction_counts = pd.Series(y_pred).value_counts()
#     prediction_data = {
#         'labels': prediction_counts.index.tolist(),
#         'data': prediction_counts.values.tolist()
#     }

#     return render(request, 'data_analysis.html', {
#         'report': {
#             'accuracy': model.score(X_test, y_test),
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1_score,
#             'confusion_matrix': cm.tolist(), 
#         },
#         'prediction_data': prediction_data
#     })

# def prediction_reports(request):
#     # Load transaction data and proceed as before
#     transactions = Transaction.objects.all()
#     transaction_df = pd.DataFrame(list(transactions.values()))

#     transaction_id_col = transaction_df['transactionID']
#     transaction_df['monthly_charges'] = pd.to_numeric(transaction_df['monthly_charges'], errors='coerce')
#     transaction_df['total_charges'] = pd.to_numeric(transaction_df['total_charges'], errors='coerce')
#     transaction_df.fillna(0, inplace=True)
#     transaction_df = clean_data(transaction_df.drop(columns=['transactionID']))
#     transaction_df['transactionID'] = transaction_id_col

#     X = transaction_df.drop(columns=['churn', 'transactionID'])
#     y = transaction_df['churn']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)

#     model_path = os.path.join(settings.BASE_DIR, 'models', 'churn_model.pkl')
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, model_path)

#     transaction_id = request.GET.get('transaction_id', None)
#     if transaction_id:
#         transaction_df = transaction_df[transaction_df['transactionID'].astype(str).str.contains(transaction_id)]

#     if transaction_df.empty:
#         return render(request, 'error.html', {'message': f'No transaction found with ID: {transaction_id}'})

#     X_test_filtered = transaction_df.drop(columns=['churn', 'transactionID'], errors='ignore')
#     predictions = dict(zip(transaction_df['transactionID'], model.predict(X_test_filtered)))

#     # Calculate prediction counts
#     prediction_counts = pd.Series(predictions.values()).value_counts()
#     prediction_data = {
#         'labels': prediction_counts.index.tolist(),
#         'data': prediction_counts.values.tolist()
#     }

#     # Pass the prediction data to the template
#     return render(request, 'prediction_reports.html', {
#         'report': {
#             'predictions': predictions,
#             'accuracy': model.score(X_test_filtered, transaction_df['churn']) if 'churn' in transaction_df.columns else 'N/A',
#             'weighted_avg': {
#                 'precision': report['weighted avg']['precision'],
#                 'recall': report['weighted avg']['recall'],
#                 'f1_score': report['weighted avg']['f1-score'],
#             }
#         },
#         'prediction_data': {
#             'labels': json.dumps(prediction_counts.index.tolist()),  # Convert to JSON string
#             'data': json.dumps(prediction_counts.values.tolist())    # Convert to JSON string
#         }
#     })
 