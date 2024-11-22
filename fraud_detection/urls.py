from django.urls import path


from .views import *

urlpatterns = [
    path('dashboard_view/', prediction_reports, name='dashboard'),
    path('data_analysis/', prediction_results, name='data_analysis'),
    path('input_transaction/', input_transaction, name='input_transaction'),
    path('upload_file/', upload_file, name='upload_file'),
    path('',transaction_list, name='transactions'),
    # path('detect-fraud/', detect_fraud, name='detect_fraud'),



]
