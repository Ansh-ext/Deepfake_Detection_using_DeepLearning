import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

IMAGE_SIZE=224
NUM_FRAMES=5

device = "cuda" if torch.cuda.is_available() else "cpu"


mtcnn = MTCNN(
    keep_all=False,
    device=device,
    post_process=False,
    min_face_size=60
)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_face_bbox(video_path, sample_frames=3):
    cap   = cv2.VideoCapture(video_path)
    print("Video opened:", cap.isOpened())
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    indices = np.linspace(0, total-1,
                          min(sample_frames, total), dtype=int)
    boxes = []
    for fidx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, _ = mtcnn.detect(Image.fromarray(rgb))
        if box is not None:
            boxes.append(box[0])
    cap.release()
    if not boxes:
        return None, h, w
    return np.mean(boxes, axis=0).astype(int), h, w


def preprocess_video(video_path, num_frames=NUM_FRAMES):
    
    bbox, h, w = get_face_bbox(video_path)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0,   x1 - 30)
        y1 = max(0,   y1 - 30)
        x2 = min(w-1, x2 + 30)
        y2 = min(h-1, y2 + 30)
    else:
        size = min(h, w)
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size

    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1,
                          min(num_frames, total), dtype=int)
    frames  = []
    raw_frames = []  

    for fidx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: continue

        crop    = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (224,224))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        raw_frames.append(rgb)
        frames.append(transform(Image.fromarray(rgb)))

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])
        raw_frames.append(raw_frames[-1])

    tensor = torch.stack(frames).unsqueeze(0).to(device)
  
    return tensor, raw_frames