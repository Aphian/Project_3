from django.db import models
from django.conf import settings
import uuid

# Create your models here.

class ImageContents(models.Model):
    # 이미지 input
    image = models.ImageField(upload_to='images/')
    
    # 이미지 uuid 생성
    image_uuid = models.TextField()

class ImageInference(models.Model):
    image_id = models.ForeignKey(ImageContents,
                                 on_delete=models.CASCADE,
                                 related_name='inf_id',
                                 )
    
    image_inf = models.ImageField(upload_to='images/')

