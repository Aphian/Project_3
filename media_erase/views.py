from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.

from uuid import uuid4
from img_erase import inference
from . models import MediaContents
from . forms import MediaContentsForm
import base64

# Create your views here.

def media_upload(request):
    pass

def inference_media(request):
    pass