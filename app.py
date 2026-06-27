import os
import tempfile

import streamlit as st

from src.inference import predict

import subprocess


def ensure_browser_compatible(input_path: str) -> str:
    """Re-encode video to H.264/yuv420p so it plays in st.video()."""
    output_path = input_path.replace(".mp4", "_web.mp4")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-acodec", "aac",
            "-movflags", "+faststart",   
            output_path,
        ],
        check=True,
        capture_output=True,
    )

    return output_path


st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "result" not in st.session_state:
    st.session_state.result = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

with st.sidebar:

    st.title("🧠 Deepfake Detector")

    st.markdown("---")

    st.markdown("""
### Model

- EfficientNet-B4
- Transformer Encoder
- YOLO Face Tracking

### Input

- 5 Frames
- 224 × 224

### Output

- REAL / FAKE
- Confidence
- Annotated Video
""")

st.title("🎥 Deepfake Detection System")

st.caption(
    "Deepfake Video Detection using CNN + Transformer"
)

st.divider()

uploaded = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov"]
)


if uploaded is not None:

    file_id = f"{uploaded.name}_{uploaded.size}"

    if file_id != st.session_state.uploaded_name:

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp.write(uploaded.read())
        temp.close()

        st.session_state.video_path = temp.name
        st.session_state.uploaded_name = file_id
        st.session_state.result = None   

    video_path = st.session_state.video_path

    left, right = st.columns([2, 1])

    with left:

        st.subheader("📹 Original Video")

        st.video(video_path)

    with right:

        st.subheader("Prediction")

        if st.button(
            "Analyze Video",
            width='stretch'
        ):

            with st.spinner("Running inference..."):

                st.session_state.result = predict(video_path)

            st.success("Analysis Complete!")

if st.session_state.result is not None:

    result = st.session_state.result

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Processed Video")

        st.video(result["annotated_video"])

        with open(result["annotated_video"], "rb") as f:

            st.download_button(
                "⬇ Download Annotated Video",
                f,
                "annotated_video.mp4",
                "video/mp4",
                width='stretch'
            )

    with right:

        st.subheader("Prediction")

        if result["label"] == "REAL":

            st.success("🟢 REAL")

        else:

            st.error("🔴 FAKE")

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )

        st.metric(
            "Fake Probability",
            f"{result['fake_prob']:.2f}%"
        )

        st.metric(
            "Real Probability",
            f"{result['real_prob']:.2f}%"
        )

    st.divider()

    st.subheader("Prediction Probability")

    c1, c2 = st.columns(2)

    with c1:

        st.write("REAL")

        st.progress(
            int(result["real_prob"])
        )

        st.write(
            f"{result['real_prob']:.2f}%"
        )

    with c2:

        st.write("FAKE")

        st.progress(
            int(result["fake_prob"])
        )

        st.write(
            f"{result['fake_prob']:.2f}%"
        )

    st.divider()

    st.subheader("Frames Used by Model")

    cols = st.columns(len(result["frames"]))

    for i, frame in enumerate(result["frames"]):

        cols[i].image(
            frame,
            caption=f"Frame {i+1}",
            width='stretch'
        )

st.divider()

st.caption(
    "© 2025 Deepfake Detection using CNN + Transformer"
)
