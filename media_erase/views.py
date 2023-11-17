from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

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

    inference_names = media_path
    inference_names = os.path.basename(media_path)
    inference_names, extension = os.path.splitext(inference_names)

    # 이미지 추론시 저장이름을 업로드 된 영상의 이름 + frame_count
    media_path = 'media/results_inference_videos'  
    before_image_names = [f for f in os.listdir(media_path) if f.startswith(inference_names) and f.endswith(('.jpg', '.png'))]

    image_names = sorted(before_image_names)

    # 페이징 설정
    paginator = Paginator(image_names, 4)  # 페이지당 8개의 항목을 표시하도록 설정
    page_number = request.GET.get('page')

    try:
        image_names = paginator.page(page_number)
    except PageNotAnInteger:
        # 페이지 번호가 정수가 아닌 경우, 첫 번째 페이지를 가져옴
        image_names = paginator.page(1)
    except EmptyPage:
        # 페이지 번호가 범위를 벗어난 경우, 마지막 페이지를 가져옴
        image_names = paginator.page(paginator.num_pages)

    
    return render(request, 'media_erase/media_inference.html', {
        'media_path' : media_path,
        # 'inference_path' : inference_path,
        'image_names': image_names,
    })