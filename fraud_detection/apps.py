# from django.apps import AppConfig


# class FraudDetectionConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'fraud_detection'

from django.apps import AppConfig
import joblib
import os
import logging

# Define the path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'fraud_detection_model.pkl')

# Set up logging
logger = logging.getLogger(__name__)

class FraudDetectionConfig(AppConfig):
    name = 'fraud_detection'

    def ready(self):
        global model
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)  # Load the model once
                logger.info(f"Model loaded successfully at startup from {model_path}")
            else:
                logger.error(f"Model file not found at: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
