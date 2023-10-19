from django.db import models
from django.conf import settings
import uuid

# Create your models here.

class DetectContents(models.Model):
    image = models.ImageField(upload_to='images/')

