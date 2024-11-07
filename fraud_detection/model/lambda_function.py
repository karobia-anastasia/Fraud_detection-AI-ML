# lambda_function.py
import json
import joblib
import pandas as pd

# Load your trained model
model = joblib.load('path/to/your/model.pkl')

def lambda_handler(event, context):
    # Assuming event is a JSON with transaction details
    transaction = json.loads(event['body'])
    df = pd.DataFrame([transaction])
    prediction = model.predict(df)

    response = {
        'statusCode': 200,
        'body': json.dumps({'transaction': transaction, 'prediction': 'Fraud' if prediction[0] else 'Legit'})
    }

    return response
