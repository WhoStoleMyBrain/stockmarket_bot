from django.urls import path
from . import views

urlpatterns = [
    path('trigger-task/', views.trigger_task, name='trigger_task'),
]
