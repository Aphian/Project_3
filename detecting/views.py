from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.

from uuid import uuid4
from . import inference
from . models import ImageContents
from . forms import ImageContentsForm
import base64

def detecting(target_img):

    target_image_path = str(target_image)
    target_image = 'media/' + target_image_path

    inference.main(target_image, target_image_path)

@ require_http_methods(['GET', 'POST'])
def main(request):
    if request.method == 'POST':
        img_form = ImageContentsForm(request.POST, request.FILES)
        if img_form.is_valid():
            image = img_form.save(commit=False)
            image.save()
            
            detecting(image.image)

            # return redirect('detecting:inference_img', uuid=image.image_uuid)
            return redirect('detecting:main')
            
    else:
        img_form = ImageContentsForm()
    
    return render(request, 'detecting/main.html', {'img_form': img_form})

def inference_image(request, uuid):
    pass

