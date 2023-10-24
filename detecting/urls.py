from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'detecting'

urlpatterns = [
    # main / image upload
    path('', views.main, name='main'),
    # inference result
    path('inference_image/', views.inference_image, name='inference_img'),

]