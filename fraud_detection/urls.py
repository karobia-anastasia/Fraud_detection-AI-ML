from django.urls import path

from .views import *

urlpatterns = [
    path('dashboard/', data_analysis_view, name='dashboard'),
    path('data_analysis/', data_analysis_view, name='data_analysis'),
    path('input_transaction/', input_transaction, name='input_transaction'),
    path('upload_file/', upload_file, name='upload_file'),
    path('',transaction_list, name='transactions'),



]
