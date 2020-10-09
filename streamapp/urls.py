from django.urls import path, include
from streamapp import views


urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('iklan_feed', views.iklan_feed, name='iklan_feed'),

]
