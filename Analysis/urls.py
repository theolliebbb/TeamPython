from django.urls import path
from . import views


urlpatterns = [
    path('graphresults', views.graphresults, name = 'graphresults'),
    path('data1', views.data1, name='data1'),
    path('comment', views.comment, name='comment'),
    path('description', views.description, name='description'),
    path('title', views.title, name='title'),
    path('videopage', views.videopage, name='videopage'),
    path('ml', views.ml.as_view(), name='ml'),
    path('', views.IndexView.as_view(), name = 'Index'),
]