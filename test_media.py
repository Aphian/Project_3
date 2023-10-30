import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video file
video_path = "input_media/video.mp4"
cap = cv2.VideoCapture(video_path)

# Define the output video writer
output_path = "output_media/output_video.mp4"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()

print(f'Result video saved at {output_path}')
