from django import forms

class TransactionForm(forms.Form):
    step = forms.IntegerField(label='Step')
    type = forms.ChoiceField(label='Transaction Type', choices=[
        ('PAYMENT', 'Payment'),
        ('TRANSFER', 'Transfer'),
        ('CASH_OUT', 'Cash Out'),
        ('CASH_IN', 'Cash In'),
    ])
    amount = forms.DecimalField(label='Amount')
    nameOrig = forms.CharField(label='Origin Name', max_length=20)
    oldbalanceOrg = forms.DecimalField(label='Old Balance Origin')
    newbalanceOrig = forms.DecimalField(label='New Balance Origin')
    nameDest = forms.CharField(label='Destination Name', max_length=20)
    oldbalanceDest = forms.DecimalField(label='Old Balance Destination')
    newbalanceDest = forms.DecimalField(label='New Balance Destination')

class UploadFileForm(forms.Form):
    file = forms.FileField()
