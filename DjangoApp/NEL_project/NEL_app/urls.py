from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='NEL_app'),
    path("run_test_on_dataset/", views.run_test_on_dataset, name="run_test_on_dataset"),
]
