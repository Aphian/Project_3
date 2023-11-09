from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.

from uuid import uuid4
from . import inference
from . models import MediaContents
from . forms import MediaContentsForm
import base64, os

# Create your views here.

def video_detect(tartget_video, media_second):
    target_video_path = str(tartget_video)
    target_video = 'media/' + target_video_path

    inference.video_inference(target_video, media_second)

@require_http_methods(['GET', 'POST'])
def media_upload(request):
    if request.method == 'POST':
        media_form = MediaContentsForm(request.POST, request.FILES)
        media_second = request.POST.get('media_second')
        if media_form.is_valid():
            media = media_form.save(commit=False)
            media.save()

            video_detect(media.media, media_second)

            return redirect('media_erase:inference_media', uuid=media.media_uuid)
    else:
        media_form = MediaContentsForm()
    return render(request, 'media_erase/upload_video.html', {
        'media_form' : media_form,
    })

def inference_media(request, uuid):
    media = get_object_or_404(MediaContents, media_uuid=uuid)
    media_path = str(media.media)
    # inference_path = media_path.replace('videos', 'results_video')

    # 이미지 추론시 저장이름을 업로드 된 영상의 이름 + frame_count
    media_path = 'media/results_inference_videos'  
    image_names = [f for f in os.listdir(media_path) if f.startswith('frame_') and f.endswith(('.jpg', '.png'))]
    
    return render(request, 'media_erase/media_inference.html', {
        'media_path' : media_path,
        # 'inference_path' : inference_path,
        'image_names': image_names,
    })
