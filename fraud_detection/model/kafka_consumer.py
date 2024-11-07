# kafka_consumer.py
from kafka import KafkaConsumer
import json
import joblib
import pandas as pd

# Load your trained model
model = joblib.load('path/to/your/model.pkl')

def predict_fraud(transaction):
    df = pd.DataFrame([transaction])
    prediction = model.predict(df)
    return prediction

def consume_transactions():
    consumer = KafkaConsumer('transactions',
                             bootstrap_servers='localhost:9092',
                             value_deserializer=lambda x: json.loads(x.decode('utf-8')))

    for message in consumer:
        transaction = message.value
        prediction = predict_fraud(transaction)
        print(f"Transaction: {transaction}, Prediction: {'Fraud' if prediction[0] else 'Legit'}")

consume_transactions()
