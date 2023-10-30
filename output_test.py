import cv2

# 동영상 파일 경로
video_path = "output_video.mp4"

# 동영상 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()

# 동영상 재생 설정
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 동영상 프레임을 표시
    # cv2.imshow("Saved Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord("q"):
        break

# 동영상 파일과 OpenCV 창 닫기
cap.release()
# cv2.destroyAllWindows()