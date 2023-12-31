from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

from uuid import uuid4
from . import inference
from . models import MediaContents
from . forms import MediaContentsForm
from . util import delete_folder_contents
import base64, os, zipfile

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
        delete_folder_contents('media/inferenced_videos')
        
        media_form = MediaContentsForm()
    return render(request, 'media_erase/upload_video.html', {
        'media_form' : media_form,
    })

def inference_media(request, uuid):
    media = get_object_or_404(MediaContents, media_uuid=uuid)
    media_path = str(media.media)
    download_name = str(media.media)

    # 파일 경로에서 파일 이름만 추출
    download_name = os.path.basename(download_name)

    # 파일 이름에서 확장자 제거
    download_name, extension = os.path.splitext(download_name)

    inference_names = media_path
    inference_names = os.path.basename(media_path)
    inference_names, extension = os.path.splitext(inference_names)

    # 이미지 추론시 저장이름을 업로드 된 영상의 이름 + frame_count
    media_path = 'media/inferenced_videos'  
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
        'download_name' : download_name,
        # 'inference_path' : inference_path,
        'image_names': image_names,
    })

@csrf_exempt
def zip_and_download(request):
    # 압축할 폴더의 경로
    folder_path = 'media/inferenced_videos/'

    # 폴더 이름 추출
    folder_name = os.path.basename(folder_path)

    # zip 파일 생성
    zip_file_path = f'downloaded_{folder_name}.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))

    # Serve the zip file with a dynamic filename
    with open(zip_file_path, 'rb') as zip_file:
        response = HttpResponse(zip_file.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="{ generate_dynamic_filename() }.zip"'
        return response
    
def generate_dynamic_filename():
    # 여기에서 동적으로 파일 이름을 생성하는 로직을 추가
    # 예: 현재 날짜와 시간을 기반으로 파일 이름 생성
    import datetime
    return f'Cleaner_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'