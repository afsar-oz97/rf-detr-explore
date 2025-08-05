import cv2
import time
import supervision as sv
from inference import get_model

# Load the RF-DETR model
model = get_model("rfdetr-base")

# Open video file (change path to your video)
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
out = cv2.VideoWriter(
    "annotated_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# Annotators
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW)
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW)

# For FPS calculation
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.infer(frame, confidence=0.5)[0]
    detections = sv.Detections.from_inference(results)
    labels = [prediction.class_name for prediction in results.predictions]

    # Annotate frame
    annotated_frame = box_annotator.annotate(frame.copy(), detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    # Write to output video
    out.write(annotated_frame)

    # FPS calculation
    frame_count += 1
    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time
    print(f"\rProcessing FPS: {current_fps:.2f}", end="")


# Release resources
cap.release()
out.release()


# Final FPS report
total_time = time.time() - start_time
final_fps = frame_count / total_time
print(
    f"\nFinished! Processed {frame_count} frames in {total_time:.2f}s ({final_fps:.2f} FPS)"
)
