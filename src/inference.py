import torch

from src.face_tracking import annotate_video
from .models import build_model
from .utils import preprocess_video


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():

    model_instance = build_model('transformer').to(device)

    checkpoint = torch.load(
        r'C:\Users\DELL - S\DeepfakeDetection\Checkpoints\best_model.pth',
        map_location=device
    )

    model_instance.load_state_dict(
        checkpoint["model_state"]
    )

    model_instance.eval()

    return model_instance

model = load_model()


def predict(video_path):
    
    print(f'Analyzing: {video_path.split("/")[-1]}')

    tensor, raw_frames = preprocess_video(video_path)

    with torch.no_grad():
        output = model(tensor)              
        prob   = torch.sigmoid(output).item()

    label      = 'FAKE' if prob > 0.5 else 'REAL'
    confidence = prob if prob > 0.5 else 1 - prob
    annotated_video = annotate_video(
                        video_path,
                        label,
                        confidence*100)

    return {
        'label'     : label,
        'confidence': confidence * 100,
        'fake_prob' : prob * 100,
        'real_prob' : (1 - prob) * 100,
        'frames'    : raw_frames,
        'annotated_video':annotated_video
    }