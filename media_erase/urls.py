from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'media_erase'

urlpatterns = [
    # media upload
    path('', views.media_upload, name='media_upload'),
    # inference result
    path('inference_media/<uuid:uuid>/', views.inference_media, name='inference_media'),
    path('zip_and_download/', views.zip_and_download, name='zip_and_download'),

]