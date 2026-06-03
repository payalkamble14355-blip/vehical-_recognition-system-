import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from ultralytics import YOLO

    # 1) Use your fine-tuned weights if present
    if os.path.exists("best.pt"):
        return YOLO("best.pt")

    # 2) Otherwise download the standard YOLOv8s classification model
    #    (works out-of-the-box on Streamlit Cloud — no manual upload needed)
    st.info("ℹ️ `best.pt` not found — loading pretrained YOLOv8s-cls as fallback.")
    return YOLO("yolov8s-cls.pt")   # ultralytics auto-downloads on first run

CLASSES      = ['Bus', 'Car', 'Motorcycle', 'Truck']
CLASS_COLORS = {'Bus': '#FF8C00', 'Car': '#2ECC71', 'Motorcycle': '#3498DB', 'Truck': '#E74C3C'}
CLASS_ICONS  = {'Bus': '🚌', 'Car': '🚗', 'Motorcycle': '🏍️', 'Truck': '🚛'}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.pred-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    text-align: center;
}
.pred-label { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; margin: 0; }
.pred-conf  { font-size: 1.1rem; color: #a0a8c0; margin-top: 4px; }
.prob-bar-container {
    background: #1a1d2e;
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid #2d3561;
    margin: 8px 0;
}
.prob-row   { display: flex; align-items: center; margin: 8px 0; gap: 10px; }
.prob-label { width: 120px; font-size: 0.9rem; color: #c8cfe8; font-weight: 500; }
.prob-track { flex: 1; height: 10px; background: #2d3561; border-radius: 5px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 5px; }
.prob-val   { width: 52px; text-align: right; font-size: 0.85rem; color: #7b84a8; font-family: 'Space Mono', monospace; }
.section-header { font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em;
                  text-transform: uppercase; color: #5a6080; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Vehicle AI")
    st.markdown("---")
    mode = st.radio("Mode", ["📷 Image", "🎬 Video", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** YOLOv8s-cls")
    st.markdown("**Classes:**")
    for cls in CLASSES:
        st.markdown(f"&nbsp;&nbsp;{CLASS_ICONS[cls]} {cls}")
    st.markdown("---")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.0, 0.05)

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict(model, img_bgr):
    gray3 = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    result = model.predict(gray3, imgsz=224, verbose=False)[0]
    probs  = result.probs.data.cpu().numpy()
    # Sync class names from model
    model_classes = [model.names[i] for i in sorted(model.names.keys())]
    return probs, model_classes

def render_prob_bars(probs, classes):
    html = '<div class="prob-bar-container"><p class="section-header">Class Probabilities</p>'
    for idx in np.argsort(probs)[::-1]:
        cls   = classes[idx] if idx < len(classes) else f"Class {idx}"
        p     = probs[idx]
        color = CLASS_COLORS.get(cls, '#7b84a8')
        icon  = CLASS_ICONS.get(cls, '🚘')
        html += f"""
        <div class="prob-row">
          <div class="prob-label">{icon} {cls}</div>
          <div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%;background:{color};"></div></div>
          <div class="prob-val">{p*100:.1f}%</div>
        </div>"""
    html += '</div>'
    return html

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Vehicle Recognition System")
st.markdown("<p style='color:#5a6080;font-family:Space Mono,monospace;font-size:0.8rem;letter-spacing:0.1em;'>YOLOV8 · BUS · CAR · MOTORCYCLE · TRUCK</p>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📷 Image":
    uploaded = st.file_uploader(
        "Upload a vehicle image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded:
        model = load_model()

        # Read image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the image. Please try a different file.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Running prediction..."):
            probs, model_classes = predict(model, img_bgr)

        top_idx  = int(np.argmax(probs))
        top_cls  = model_classes[top_idx] if top_idx < len(model_classes) else f"Class {top_idx}"
        top_conf = float(probs[top_idx])

        # Map model classes → display classes for colours/icons
        display_cls = top_cls if top_cls in CLASS_COLORS else top_cls

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<p class="section-header">Input Image</p>', unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.markdown('<p class="section-header">Prediction</p>', unsafe_allow_html=True)

            if top_conf >= conf_threshold:
                color = CLASS_COLORS.get(display_cls, '#4c72b0')
                icon  = CLASS_ICONS.get(display_cls, '🚘')
                st.markdown(f"""
                <div class="pred-card">
                    <p style="font-size:3rem;margin:0">{icon}</p>
                    <p class="pred-label" style="color:{color}">{display_cls}</p>
                    <p class="pred-conf">{top_conf*100:.1f}% confidence</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning(f"Top prediction ({top_conf*100:.1f}%) is below the confidence threshold ({conf_threshold*100:.0f}%). Lower the slider on the left.")

            st.markdown(render_prob_bars(probs, model_classes), unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3a4060;">
            <p style="font-size:3rem">📷</p>
            <p style="font-family:'Space Mono',monospace;font-size:0.9rem;letter-spacing:0.1em;">UPLOAD AN IMAGE TO BEGIN</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🎬 Video":
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video:
        model = load_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

        output_path = tmp_in_path.replace(".mp4", "_output.mp4")
        cap   = cv2.VideoCapture(tmp_in_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out   = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        COLORS_BGR = {'Bus':(50,140,255),'Car':(50,200,50),'Motorcycle':(219,152,52),'Truck':(52,94,235)}
        FONT       = cv2.FONT_HERSHEY_DUPLEX

        progress = st.progress(0)
        status   = st.empty()
        count    = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            gray3  = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            result = model.predict(gray3, imgsz=224, verbose=False)[0]
            probs_arr = result.probs.data.cpu().numpy()
            pred_idx  = int(probs_arr.argmax())
            pred_conf = float(probs_arr.max())
            pred_name = model.names.get(pred_idx, str(pred_idx))
            label     = f"{pred_name}  {pred_conf*100:.1f}%"
            color     = COLORS_BGR.get(pred_name, (200, 200, 200))
            cv2.putText(frame, label, (20, 50), FONT, 1.2, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (20, 50), FONT, 1.2, color,   2, cv2.LINE_AA)
            out.write(frame)
            if count % 10 == 0:
                progress.progress(min(count / max(total, 1), 1.0))
                status.text(f"Frame {count}/{total}")

        cap.release()
        out.release()
        os.unlink(tmp_in_path)
        progress.progress(1.0)
        status.text(f"✅ Done! Processed {count} frames.")

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Processed Video", f,
                               file_name="vehicle_output.mp4", mime="video/mp4",
                               use_container_width=True)
        os.unlink(output_path)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3a4060;">
            <p style="font-size:3rem">🎬</p>
            <p style="font-family:'Space Mono',monospace;font-size:0.9rem;letter-spacing:0.1em;">UPLOAD A VIDEO TO PROCESS</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT MODE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "ℹ️ About":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Project Summary")
        st.markdown("""
        This system uses **YOLOv8s-cls** fine-tuned on a vehicle dataset to classify
        images and video frames into four categories.

        **Pipeline:**
        1. Image upload → grayscale preprocessing
        2. YOLOv8 classification inference (224×224)
        3. Softmax probabilities → top prediction displayed
        """)
    with col2:
        st.markdown("### Model Details")
        for k, v in {
            "Architecture": "YOLOv8s-cls",
            "Input size": "224 × 224 px",
            "Classes": "Bus, Car, Motorcycle, Truck",
            "Dataset": "~8,863 images",
            "Split": "70% / 15% / 15%",
            "Preprocessing": "Grayscale normalisation",
        }.items():
            st.markdown(f"**{k}:** {v}")

    st.markdown("### Technologies")
    for col, t in zip(st.columns(4), ["Python", "YOLOv8", "OpenCV", "Streamlit"]):
        col.markdown(f"""
        <div style="background:#1a1d2e;border:1px solid #2d3561;border-radius:10px;
                    padding:16px;text-align:center;font-family:'Space Mono',monospace;
                    font-size:0.85rem;color:#c8cfe8;">{t}</div>""", unsafe_allow_html=True)
