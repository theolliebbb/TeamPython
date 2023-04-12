from django.urls import path
from . import views


urlpatterns = [
    path('graphresults', views.graphresults, name = 'graphresults'),
    path('data1', views.data1, name='data1'),
    path('', views.IndexView.as_view(), name = 'Index'),
]