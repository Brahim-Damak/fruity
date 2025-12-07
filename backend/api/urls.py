from django.urls import path
from . import views

urlpatterns = [
    path('info/', views.api_info, name='api_info'),
    path('predict/', views.predict_vegetable, name='predict'),
    path('predictions/', views.get_predictions, name='predictions_list'),
    path('predictions/<int:pk>/', views.get_prediction_detail, name='prediction_detail'),
]
   