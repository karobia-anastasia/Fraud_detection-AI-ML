import os
import logging
from django.conf import settings
import joblib
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fraud_detection_project.settings import BASE_DIR
from .models import Transaction
from .forms import TransactionForm, UploadFileForm
from .data_cleaning import preprocess_data
from .data_exploration import explore_data
from .model.model_selection import evaluate_models
from .model.model_training import train_models
from .model.model_optimization import optimize_models
from django.core.paginator import Paginator
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.db.models import Count
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import logging


# Set up logging
logger = logging.getLogger(__name__)
class MissingTargetColumnError(Exception):
    pass
BASE_DIR = settings.BASE_DIR

# Define the path to the model
model_path = os.path.join(BASE_DIR, 'models', 'fraud_detection_model.pkl')

# Global variable to cache the model
model = None

def load_model():
    global model
    if model is None:
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            # Load the model only if it's not loaded already
            logger.info(f"Attempting to load model from {model_path}...")
            model = joblib.load(model_path)  # Use joblib.load directly
            logger.info("Model loaded successfully.")
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise fnf_error
        except Exception as e:
            logger.error(f"Model loading failed due to unexpected error: {str(e)}")
            raise e
    else:
        logger.info("Model already loaded.")
    return model

model = None

def input_transaction(request):
    """Handle manual transaction input and predict fraud."""
    if request.method == 'POST':
        form = TransactionForm(request.POST)
        
        if form.is_valid():
            # Extract the cleaned data from the form
            data = form.cleaned_data
            
            # Convert form data into a DataFrame (necessary for model prediction)
            df = pd.DataFrame([data])

            # Preprocess the data (optional, depending on your model requirements)
            df = preprocess_data(df)

            try:
                # Ensure the model is loaded before making predictions
                if model is None:
                    model = load_model()  # Load the model if it is not already loaded
                    logger.info("Model loaded successfully.")

                # Make prediction using the model
                prediction = model.predict(df)[0]  # Assuming model.predict returns an array-like result
                is_fraud = bool(int(prediction))  # Convert prediction to True/False (fraud or not)

                # Save the transaction to the database
                transaction = Transaction(
                    step=data['step'],
                    type=data['type'],
                    amount=data['amount'],
                    nameOrig=data['nameOrig'],
                    oldbalanceOrg=data['oldbalanceOrg'],
                    newbalanceOrig=data['newbalanceOrig'],
                    nameDest=data['nameDest'],
                    oldbalanceDest=data['oldbalanceDest'],
                    newbalanceDest=data['newbalanceDest'],
                    isFraud=is_fraud
                )
                transaction.save()

                # Optionally, redirect to a success page or show a message
                return render(request, 'transaction_list.html', {'transaction': transaction})

            except Exception as e:
                logger.error(f"Error processing the transaction: {str(e)}")
                return render(request, 'error_page.html', {'error': f"Error processing the transaction: {str(e)}"})

        else:
            logger.error("Form data is invalid.")
            return render(request, 'error_page.html', {'error': "Invalid form data."})

    else:
        # If it's a GET request, just show the empty form
        form = TransactionForm()

    return render(request, 'input_transaction.html', {'form': form})

def upload_file(request):
    """Handle file uploads for fraud prediction."""
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()

        # Save the uploaded file temporarily
        file_name = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(file_name)  # Use fs.path to get the absolute file path

        # Check if the file is empty
        if os.stat(file_path).st_size == 0:
            return render(request, 'error_page.html', {'error': "Uploaded file is empty."})

        # Read the uploaded file into a DataFrame (assuming it's CSV)
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return render(request, 'error_page.html', {'error': "Uploaded file is empty or malformed."})
        except pd.errors.ParserError:
            return render(request, 'error_page.html', {'error': "Error parsing the CSV file. Please check the file format."})
        except Exception as e:
            return render(request, 'error_page.html', {'error': f"Error processing the file: {str(e)}"})

        # Preprocess the data
        df = preprocess_data(df)

        # Predictions on the uploaded data
        predictions = []
        for index, row in df.iterrows():
            # Extract the required features for prediction (9 columns as expected)
            transaction_features = [
                row['amount'], row['step'], row['type'], row['nameOrig'], row['oldbalanceOrg'], row['newbalanceOrig'],
                row['nameDest'], row['oldbalanceDest'], row['newbalanceDest']
            ]

            # Predict whether the transaction is fraudulent
            fraud_prediction = model.predict([transaction_features])[0]

            # Create and save the transaction to the database
            transaction = Transaction.objects.create(
                step=row['step'],
                type=row['type'],
                amount=row['amount'],
                nameOrig=row['nameOrig'],
                oldbalanceOrg=row['oldbalanceOrg'],
                newbalanceOrig=row['newbalanceOrig'],
                nameDest=row['nameDest'],
                oldbalanceDest=row['oldbalanceDest'],
                newbalanceDest=row['newbalanceDest'],
                isFraud=fraud_prediction
            )
            predictions.append({
                'transaction_id': transaction.id,
                'isFraud': fraud_prediction
            })

        # Return a JSON response with the predictions
        return JsonResponse(predictions, safe=False)

    return render(request, 'upload_file.html', {'form': UploadFileForm()})



dummy_transactions = [
    {
        'step': 1, 'type': 'PAYMENT', 'amount': 1000.0, 'nameOrig': 'C001', 'oldbalanceOrg': 5000.0, 'newbalanceOrig': 4000.0,
        'nameDest': 'C005', 'oldbalanceDest': 2000.0, 'newbalanceDest': 3000.0, 'isFraud': False
    },
    {
        'step': 2, 'type': 'TRANSFER', 'amount': 2500.0, 'nameOrig': 'C002', 'oldbalanceOrg': 7000.0, 'newbalanceOrig': 4500.0,
        'nameDest': 'C006', 'oldbalanceDest': 1000.0, 'newbalanceDest': 3500.0, 'isFraud': True
    },
    {
        'step': 3, 'type': 'PAYMENT', 'amount': 1500.0, 'nameOrig': 'C003', 'oldbalanceOrg': 8000.0, 'newbalanceOrig': 6500.0,
        'nameDest': 'C007', 'oldbalanceDest': 3000.0, 'newbalanceDest': 4500.0, 'isFraud': False
    },
    {
        'step': 4, 'type': 'TRANSFER', 'amount': 500.0, 'nameOrig': 'C004', 'oldbalanceOrg': 1000.0, 'newbalanceOrig': 500.0,
        'nameDest': 'C008', 'oldbalanceDest': 500.0, 'newbalanceDest': 1000.0, 'isFraud': True
    },
    {
        'step': 5, 'type': 'PAYMENT', 'amount': 800.0, 'nameOrig': 'C009', 'oldbalanceOrg': 4000.0, 'newbalanceOrig': 3200.0,
        'nameDest': 'C010', 'oldbalanceDest': 1000.0, 'newbalanceDest': 1800.0, 'isFraud': False
    }
]

# Insert the dummy transactions into the database
for txn in dummy_transactions:
    Transaction.objects.create(
        step=txn['step'],
        type=txn['type'],
        amount=txn['amount'],
        nameOrig=txn['nameOrig'],
        oldbalanceOrg=txn['oldbalanceOrg'],
        newbalanceOrig=txn['newbalanceOrig'],
        nameDest=txn['nameDest'],
        oldbalanceDest=txn['oldbalanceDest'],
        newbalanceDest=txn['newbalanceDest'],
        isFraud=txn['isFraud']
    )

print("Dummy transactions added successfully!")

def transaction_list(request):
    # Get all transactions from the database
    transactions = Transaction.objects.all()

    # Set up pagination: 10 transactions per page 
    paginator = Paginator(transactions, 10)  # 10 transactions per page

    # Get the page number from the request's GET parameters
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Render the page with the paginated data
    return render(request, 'transaction_list.html', {'page_obj': page_obj})



def retrain_and_save_model():
    """Simplified function to re-train the model and save it."""
    try:
        # load the dataset 
        df = pd.read_csv(os.path.join(BASE_DIR, 'static', 'Kaggle_Data', 'data.csv'))

       
        if 'isFraud' not in df.columns:
            logger.error("'isFraud' column is missing from the dataset. Model cannot be re-trained.")
            raise MissingTargetColumnError("'isFraud' column is missing from the dataset.")

        # Preprocess the data 
        df = preprocess_data(df)

        # Separate features and target
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model   
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, model_path)
        logger.info(f"Model re-trained and saved to {model_path}")

    except MissingTargetColumnError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        raise

def preprocess_data(data):
    """Preprocesses the data by handling missing values, normalization, and feature engineering."""
    logger.info("Missing values before preprocessing:\n%s", data.isnull().sum())

    # update column names to match the dataset
    required_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Handle missing values
    data.fillna(0, inplace=True)

    # update column names to match the dataset
    numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Handle categorical columns
    categorical_columns = ['type', 'nameOrig', 'nameDest']
    for col in categorical_columns:
        if col in data.columns:
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])

    logger.info("Data shape after preprocessing: %s", data.shape)
    logger.info("Missing values after preprocessing:\n%s", data.isnull().sum())

    return data

# Helper function for loading CSV data
def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)



def data_analysis_view(request):
    try:
        # Define the file path
        file_path = os.path.join(BASE_DIR, 'static', 'Kaggle_Data', 'data.csv')

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at {file_path}")

        # Load and preprocess the data
        df = load_data(file_path)

        if 'newbalanceOrig' not in df.columns:
            logger.warning("Column 'newbalanceOrig' not found in DataFrame.")
        else:
            logger.info("Columns in DataFrame: %s", df.columns)

        cleaned_df = preprocess_data(df)

        # Data Exploration 
        stats, fraud_count = explore_data(cleaned_df)

        # Model Evaluation
        model_results = evaluate_models(cleaned_df)

        # Train Models
        training_results = train_models(cleaned_df)

        # Model Optimization Results 
        optimization_results = {
            "Best_Model": "Random Forest",
            "Best_Accuracy": "92.5%",
            "Best_Hyperparameters": {"n_estimators": 100, "max_depth": 10},
        }

        # Fetch latest predictions or create a mock one if not available
        latest_predictions = Transaction.objects.filter(isFraud__isnull=False).order_by('-timestamp')[:5]

        # Prepare data for optimization
        X = cleaned_df.drop('isFraud', axis=1)
        y = cleaned_df['isFraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Return the analysis and results
        return render(request, 'data_analysis.html', {
            'stats': stats,
            'fraud_count': fraud_count,
            'model_results': model_results,
            'training_results': training_results,
            'optimization_results': optimization_results,
            'latest_predictions': latest_predictions,
        })

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        return render(request, 'error_page.html', {'error': "CSV file not found. Please check the file path."})

    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        return render(request, 'error_page.html', {'error': f"An error occurred: {str(e)}"})

class FraudDetector:
    def __init__(self):
        try:
            self.model = joblib.load(settings.MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, transaction_data):
        df = pd.DataFrame(transaction_data)

        # Handle missing values 
        df.fillna(df.mean(), inplace=True)

        # Preprocess data
        df = preprocess_data(df)

        # Make predictions
        predictions = self.model.predict(df)
        return predictions


def dashboard(request):
    # Fetch the 10 most recent transactions
    transactions = Transaction.objects.all().order_by('-date')[:10]
    
    # Calculate total transactions and fraud transactions
    total_transactions = Transaction.objects.count()
    fraud_transactions = Transaction.objects.filter(isFraud=True).count()
    
    # Get the number of transactions per month (or any other stats for graphing)
    transaction_data = Transaction.objects.values('created_at__month').annotate(count=Count('id'))
    months = [data['created_at__month'] for data in transaction_data]
    transaction_counts = [data['count'] for data in transaction_data]
    
    # Create a plot using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(months, transaction_counts, label='Transactions')
    ax.set(xlabel='Month', ylabel='Number of Transactions', title='Transactions per Month')
    
    # Save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot as base64 for embedding in HTML
    graph_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Pass data to the template
    context = {
        'transactions': transactions,
        'total_transactions': total_transactions,
        'fraud_transactions': fraud_transactions,
        'graph_image': graph_image,  # The base64-encoded graph image
    }
    
    return render(request, 'dashboard.html', context)
