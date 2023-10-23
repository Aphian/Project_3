from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.

from uuid import uuid4
from . import inference
from . models import ImageContents
from . forms import ImageContentsForm
import base64

def rename_imagefile_to_uuid(filename): 
    ext = filename.split('.')[-1] 
    uuid = uuid4().hex
    filename = '{}.{}'.format(uuid, ext)
    return filename

def upload_image(request):
    if request.method == 'POST':
        img_form = ImageContentsForm(request.POST, request.FILES)
        if img_form.is_valid():
            image = img_form.save(commit=False)
            image.save()

            
