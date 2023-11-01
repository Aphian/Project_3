from django.db import models
from django.conf import settings
import uuid

# Create your models here.

class ImageContents(models.Model):
    # 이미지 input
    image = models.ImageField(upload_to='images/')
    
    # 이미지 uuid 생성
    image_uuid = models.UUIDField(unique=True, default=uuid.uuid4, editable=False)

