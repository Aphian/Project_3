import cv2
import os

# 동영상 파일 경로
video_path = 'output_video.mp4'

# 이미지를 저장할 디렉토리 경로
output_directory = 'frame_save/'

# 만약 디렉토리가 존재하지 않으면 생성
os.makedirs(output_directory, exist_ok=True)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 프레임 간격
frame_interval = 10

# 프레임 수 초기화
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 각 프레임을 이미지로 저장
    frame_count += 1
    if frame_count % frame_interval == 0:
        image_filename = os.path.join(output_directory, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(image_filename, frame)
        
# 비디오 캡처 객체와 창 닫기
cap.release()
# cv2.destroyAllWindows()
