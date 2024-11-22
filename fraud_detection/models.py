import joblib
import pandas as pd
from django.conf import settings
from django.db import models

class Transaction(models.Model):
    STEP_CHOICES = [(i, i) for i in range(1, 31)]
    TRANSACTION_TYPES = [
        ('PAYMENT', 'Payment'),
        ('TRANSFER', 'Transfer'),
        ('CASH_OUT', 'Cash Out'),
    ]

    step = models.IntegerField(choices=STEP_CHOICES, default=1)
    type = models.CharField(max_length=10, choices=TRANSACTION_TYPES)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    nameOrig = models.CharField(max_length=20)
    oldbalanceOrg = models.DecimalField(max_digits=15, decimal_places=2)
    newbalanceOrig = models.DecimalField(max_digits=15, decimal_places=2)
    nameDest = models.CharField(max_length=20)
    oldbalanceDest = models.DecimalField(max_digits=15, decimal_places=2)
    newbalanceDest = models.DecimalField(max_digits=15, decimal_places=2)
    isFraud = models.BooleanField(default=False)
    isFlaggedFraud = models.BooleanField(default=False)
    prediction = models.IntegerField(null=True, blank=True)
    prediction_label = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"Transaction {self.id} - {self.type} - Fraud: {self.isFraud}"


