import os
import cv2
from ultralytics import YOLO

# Load YOLO Face model once
face_model = YOLO("weights/yolov8n-face.pt")


def annotate_video(
    video_path,
    label,
    confidence,
    output_path="outputs/annotated_video.mp4"
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (width, height)
    )

    color = (0,255,0) if label=="REAL" else (0,0,255)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO Face Detection
        results = face_model(frame, verbose=False)

        if len(results):

            boxes = results[0].boxes

            if len(boxes):

                # Largest face
                largest = max(
                    boxes,
                    key=lambda b:
                    (b.xyxy[0][2]-b.xyxy[0][0])*
                    (b.xyxy[0][3]-b.xyxy[0][1])
                )

                x1,y1,x2,y2 = map(
                    int,
                    largest.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    color,
                    3
                )

                cv2.putText(
                    frame,
                    f"{label} {confidence:.1f}%",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        writer.write(frame)

    cap.release()
    writer.release()

    return output_path