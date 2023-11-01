from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.

from uuid import uuid4
from . import inference
from . models import MediaContents
from . forms import MediaContentsForm
import base64

# Create your views here.

def video_detect(tartget_video):
    target_video_path = str(tartget_video)
    target_video = 'media/' + target_video_path

    inference.video_inference(target_video, target_video_path)

@require_http_methods(['GET', 'POST'])
def media_upload(request):
    if request.method == 'POST':
        media_form = MediaContentsForm(request.POST, request.FILES)
        if media_form.is_valid():
            media = media_form(commit=False)
            media.save()

            video_detect(media.media)

            return redirect('media_erase:inference_media', uuid=media.media_uuid)
    else:
        media_form = MediaContentsForm()
    return render(request, 'media_erase/upload_video.html', {
        'media_form' : media_form,
    })

def inference_media(request):
    pass