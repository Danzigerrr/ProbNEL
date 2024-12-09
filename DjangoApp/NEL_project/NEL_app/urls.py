from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='NEL_app'),  # This handles the 'NEL_app/' route
]
