import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model file `best.pt` not found. Please upload it to the app directory.")
        st.stop()
    return YOLO(model_path)

CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']
CLASS_COLORS = {
    'Bus':        '#00F5FF',   # electric cyan
    'Car':        '#BF00FF',   # neon purple
    'Motorcycle': '#FF2079',   # hot magenta
    'Truck':      '#00FF9F',   # neon green
}
CLASS_ICONS = {
    'Bus': '🚌',
    'Car': '🚗',
    'Motorcycle': '🏍️',
    'Truck': '🚛',
}

# ── Plot theme helper ──────────────────────────────────────────────────────────
PLT_BG       = "#05010F"
PLT_SURFACE  = "#0D0720"
PLT_BORDER   = "#2A0A5E"
PLT_TEXT     = "#C8B8FF"
PLT_SUBTEXT  = "#4A3580"

CURVE_COLORS = {
    "train": "#00F5FF",
    "val":   "#BF00FF",
    "top1":  "#00FF9F",
    "top3":  "#FF2079",
    "bus":   "#00F5FF",
    "car":   "#BF00FF",
    "motorcycle": "#FF2079",
    "truck": "#00FF9F",
}

def apply_dark_theme(fig, ax_list):
    fig.patch.set_facecolor(PLT_BG)
    for ax in ax_list:
        ax.set_facecolor(PLT_SURFACE)
        ax.tick_params(colors=PLT_TEXT, labelsize=9)
        ax.xaxis.label.set_color(PLT_TEXT)
        ax.yaxis.label.set_color(PLT_TEXT)
        ax.title.set_color(PLT_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLT_BORDER)
        ax.grid(color=PLT_BORDER, linestyle="--", linewidth=0.6, alpha=0.6)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; letter-spacing: 0.06em; }

    .main { background-color: #05010F; }
    .stApp {
        background-color: #05010F;
        background-image:
            linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    /* ── Prediction card ── */
    .pred-card {
        background: linear-gradient(135deg, #0D0720 0%, #05010F 100%);
        border: 1px solid #2A0A5E;
        border-radius: 4px;
        padding: 28px 24px;
        margin: 12px 0;
        text-align: center;
        animation: fadeSlideIn 0.4s ease;
        box-shadow: 0 0 30px rgba(0,245,255,0.08), inset 0 0 60px rgba(191,0,255,0.04);
        position: relative;
        overflow: hidden;
    }
    .pred-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 60%; height: 1px;
        background: linear-gradient(90deg, transparent, #00F5FF, transparent);
        animation: scanline 3s linear infinite;
    }
    @keyframes scanline {
        0% { left: -60%; } 100% { left: 160%; }
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .pred-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .pred-conf {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1rem;
        color: #4A3580;
        margin-top: 6px;
        letter-spacing: 0.06em;
    }

    /* ── Probability bars ── */
    .prob-bar-container {
        background: #0D0720;
        border-radius: 4px;
        padding: 18px 22px;
        border: 1px solid #2A0A5E;
        margin: 8px 0;
        box-shadow: inset 0 0 40px rgba(191,0,255,0.04);
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 8px 0;
        gap: 10px;
    }
    .prob-label {
        width: 120px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        color: #C8B8FF;
    }
    .prob-track {
        flex: 1;
        height: 6px;
        background: #1A0A35;
        border-radius: 3px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
        box-shadow: 0 0 8px currentColor;
    }
    .prob-val {
        width: 48px;
        text-align: right;
        font-size: 0.82rem;
        color: #4A3580;
        font-family: 'Share Tech Mono', monospace;
    }

    /* ── Section header ── */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.62rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #4A3580;
        margin-bottom: 12px;
        padding-left: 8px;
        border-left: 2px solid #BF00FF;
    }

    /* ── Batch result card ── */
    .batch-card {
        background: #0D0720;
        border: 1px solid #2A0A5E;
        border-radius: 4px;
        padding: 14px 12px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .batch-card:hover {
        transform: translateY(-3px);
        border-color: #00F5FF;
        box-shadow: 0 0 20px rgba(0,245,255,0.15);
    }
    .batch-card-icon { font-size: 2rem; margin-bottom: 6px; }
    .batch-card-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .batch-card-conf {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: #4A3580;
        margin-top: 2px;
    }
    .batch-card-filename {
        font-size: 0.68rem;
        color: #2A0A5E;
        margin-top: 4px;
        font-family: 'Share Tech Mono', monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Summary stat box ── */
    .stat-box {
        background: #0D0720;
        border: 1px solid #2A0A5E;
        border-radius: 4px;
        padding: 16px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .stat-box::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #BF00FF55, transparent);
    }
    .stat-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
        text-shadow: 0 0 20px currentColor;
    }
    .stat-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.68rem;
        color: #4A3580;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* ── Upload area ── */
    div[data-testid="stFileUploader"] {
        background: #0D0720;
        border: 1px dashed #2A0A5E;
        border-radius: 4px;
        padding: 10px;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #00F5FF;
        box-shadow: 0 0 20px rgba(0,245,255,0.08);
    }

    /* ── Dataset sample card ── */
    .ds-card {
        background: #0D0720;
        border: 1px solid #2A0A5E;
        border-radius: 4px;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .ds-card:hover {
        transform: translateY(-3px);
        border-color: #00F5FF;
        box-shadow: 0 0 20px rgba(0,245,255,0.12);
    }
    .ds-card-label {
        padding: 8px 10px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .ds-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 2px;
        font-size: 0.68rem;
        font-family: 'Share Tech Mono', monospace;
        font-weight: 700;
    }

    /* ── Metric card (about page) ── */
    .tech-chip {
        background: #0D0720;
        border: 1px solid #2A0A5E;
        border-radius: 4px;
        padding: 16px;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.72rem;
        color: #00F5FF;
        letter-spacing: 0.1em;
        text-shadow: 0 0 12px rgba(0,245,255,0.5);
        transition: box-shadow 0.3s, border-color 0.3s;
    }
    .tech-chip:hover {
        box-shadow: 0 0 24px rgba(0,245,255,0.2);
        border-color: #00F5FF;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #080415 !important;
        border-right: 1px solid #2A0A5E;
    }

    /* ── HR ── */
    hr { border-color: #2A0A5E !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; background: #05010F; }
    ::-webkit-scrollbar-thumb { background: #2A0A5E; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #BF00FF; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-family:Orbitron,sans-serif;font-size:0.9rem;letter-spacing:0.14em;color:#00F5FF;text-shadow:0 0 16px rgba(0,245,255,0.7);margin-bottom:2px;text-transform:uppercase;'>VEHICLE.AI</p><p style='font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:0.2em;color:#4A3580;margin-top:0;'>RECOGNITION SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio(
        "Mode",
        ["🏠 Home", "📷 Image", "🖼️ Batch", "📈 Training Curves", "🗂️ Dataset", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#4A3580;'>MODEL:</span> <span style='color:#BF00FF;font-family:Share Tech Mono,monospace;font-size:0.8rem;text-shadow:0 0 8px rgba(191,0,255,0.6);'>YOLOv8s-cls</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#4A3580;'>CLASSES:</span>", unsafe_allow_html=True)
    for cls in CLASSES:
        st.markdown(f"&nbsp;&nbsp;{CLASS_ICONS[cls]} {cls}")
    st.markdown("---")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.0, 0.05)

# ── Helper: run inference ──────────────────────────────────────────────────────
def predict(model, img_bgr):
    gray3 = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    result = model.predict(gray3, imgsz=224, verbose=False)[0]
    probs = result.probs.data.cpu().numpy()
    model_classes = [model.names[i] for i in sorted(model.names.keys())]
    return probs, model_classes

def render_prob_bars(probs, classes):
    bars_html = '<div class="prob-bar-container"><p class="section-header">Class Probabilities</p>'
    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        cls = classes[idx]
        p = probs[idx]
        color = CLASS_COLORS.get(cls, '#4A3580')
        icon = CLASS_ICONS.get(cls, '🚘')
        bars_html += f"""
        <div class="prob-row">
            <div class="prob-label">{icon} {cls}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{p*100:.1f}%; background:{color};"></div>
            </div>
            <div class="prob-val">{p*100:.1f}%</div>
        </div>"""
    bars_html += '</div>'
    return bars_html

# ── Main title ─────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-family:Orbitron,sans-serif;font-size:1.7rem;font-weight:900;color:#00F5FF;letter-spacing:0.1em;text-shadow:0 0 24px rgba(0,245,255,0.6);margin-bottom:2px;'>VEHICLE RECOGNITION</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family:Share Tech Mono,monospace;color:#4A3580;font-size:0.72rem;letter-spacing:0.22em;'>YOLOV8 &nbsp;//&nbsp; BUS &nbsp;//&nbsp; CAR &nbsp;//&nbsp; MOTORCYCLE &nbsp;//&nbsp; TRUCK</p>", unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🏠 Home":
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0D0720 0%, #05010F 60%, #0A0020 100%);
        border: 1px solid #2A0A5E;
        border-radius: 6px;
        padding: 52px 48px 44px;
        text-align: center;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -40px; left: -40px;
            width: 200px; height: 200px;
            background: radial-gradient(circle, #00F5FF18 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute; bottom: -40px; right: -40px;
            width: 200px; height: 200px;
            background: radial-gradient(circle, #BF00FF18 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#00F5FF,transparent);"></div>
        <div style="position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#BF00FF,transparent);"></div>
        <p style="font-size:4rem; margin:0 0 12px 0; line-height:1; filter:drop-shadow(0 0 12px #00F5FF);">🚗</p>
        <h1 style="
            font-family:'Orbitron',sans-serif;
            font-size:2.2rem;
            font-weight:900;
            color:#00F5FF;
            margin:0 0 10px 0;
            letter-spacing:0.1em;
            text-shadow: 0 0 30px rgba(0,245,255,0.6), 0 0 60px rgba(0,245,255,0.2);
            text-transform:uppercase;
        ">Vehicle Recognition System</h1>
        <p style="
            color:#4A3580;
            font-family:'Share Tech Mono',monospace;
            font-size:0.72rem;
            letter-spacing:0.22em;
            margin:0 0 24px 0;
        ">YOLOV8S-CLS &nbsp;//&nbsp; BUS &nbsp;//&nbsp; CAR &nbsp;//&nbsp; MOTORCYCLE &nbsp;//&nbsp; TRUCK</p>
        <p style="color:#C8B8FF; font-family:'Rajdhani',sans-serif; font-size:1.05rem; max-width:560px; margin:0 auto; line-height:1.7;">
            A deep learning system that classifies vehicle images into four categories
            in real time using a fine-tuned YOLOv8s classification model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-stats row ───────────────────────────────────────────────────────
    qs = st.columns(4)
    quick_stats = [
        ("8,863",  "Total Images",    "#00F5FF"),
        ("4",      "Classes",         "#BF00FF"),
        ("224px",  "Input Size",      "#FF2079"),
        ("70/15/15", "Train/Val/Test", "#00FF9F"),
    ]
    for col, (val, lbl, color) in zip(qs, quick_stats):
        col.markdown(f"""
        <div class="stat-box" style="border-color:{color}44;">
            <div class="stat-number" style="color:{color}; font-size:1.6rem;">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">What you can do</p>', unsafe_allow_html=True)
    fc = st.columns(4)
    features = [
        ("📷", "Image",          "Upload a single image and get an instant classification with confidence scores and probability bars.", "#00F5FF"),
        ("🖼️", "Batch",          "Predict on multiple images at once. Export results to CSV and see class distribution at a glance.",   "#BF00FF"),
        ("📈", "Training Curves","Upload your YOLOv8 results.csv to visualise loss, accuracy, and learning-rate curves over epochs.",    "#FF2079"),
        ("🗂️", "Dataset",        "Explore the dataset composition — class counts, train/val/test splits, and proportion charts.",        "#00FF9F"),
    ]
    for col, (icon, title, desc, color) in zip(fc, features):
        col.markdown(f"""
        <div style="
            background: #0D0720;
            border: 1px solid #2A0A5E;
            border-top: 2px solid {color};
            border-radius: 4px;
            padding: 22px 18px;
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 0 0 transparent;
        ">
            <p style="font-size:1.8rem; margin:0 0 10px 0; filter:drop-shadow(0 0 8px {color});">{icon}</p>
            <p style="
                font-family:'Orbitron',sans-serif;
                font-size:0.72rem;
                font-weight:700;
                color:{color};
                margin:0 0 8px 0;
                letter-spacing:0.1em;
                text-transform:uppercase;
                text-shadow:0 0 12px {color}88;
            ">{title}</p>
            <p style="color:#4A3580; font-family:'Rajdhani',sans-serif; font-size:0.95rem; line-height:1.6; margin:0;">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Class showcase ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Supported Classes</p>', unsafe_allow_html=True)
    cc = st.columns(4)
    class_descs = {
        'Bus':        "City buses, coaches, school buses, and other large passenger transit vehicles.",
        'Car':        "Sedans, SUVs, hatchbacks, coupes — standard four-wheeled passenger cars.",
        'Motorcycle': "Two-wheelers including motorbikes, scooters, and sport motorcycles.",
        'Truck':      "Pickup trucks, lorries, freight trucks, and other heavy commercial vehicles.",
    }
    for col, cls in zip(cc, CLASSES):
        color = CLASS_COLORS[cls]
        icon  = CLASS_ICONS[cls]
        col.markdown(f"""
        <div style="
            background:#0D0720;
            border:1px solid #2A0A5E;
            border-top:2px solid {color};
            border-radius:4px;
            padding:20px 16px;
            text-align:center;
        ">
            <p style="font-size:2.4rem; margin:0 0 8px 0; filter:drop-shadow(0 0 10px {color});">{icon}</p>
            <p style="
                font-family:'Orbitron',sans-serif;
                font-size:0.75rem;
                font-weight:700;
                color:{color};
                margin:0 0 8px 0;
                letter-spacing:0.1em;
                text-shadow:0 0 12px {color}88;
                text-transform:uppercase;
            ">{cls}</p>
            <p style="color:#4A3580; font-family:'Rajdhani',sans-serif; font-size:0.9rem; line-height:1.55; margin:0;">{class_descs[cls]}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">How it works</p>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload",     "Drop an image or batch of images into the app.",              "#00F5FF"),
        ("02", "Preprocess", "Frames are converted to grayscale and resized to 224×224.",   "#BF00FF"),
        ("03", "Infer",      "YOLOv8s-cls runs a forward pass and outputs class logits.",   "#FF2079"),
        ("04", "Result",     "Softmax probabilities are ranked and displayed with labels.", "#00FF9F"),
    ]
    pipe_cols = st.columns(len(steps))
    for col, (num, title, desc, color) in zip(pipe_cols, steps):
        col.markdown(f"""
        <div style="
            background:#0D0720;
            border:1px solid #2A0A5E;
            border-radius:4px;
            padding:18px 14px;
            text-align:center;
        ">
            <p style="
                font-family:'Share Tech Mono',monospace;
                font-size:1.8rem;
                font-weight:700;
                color:{color}22;
                margin:0 0 6px 0;
                line-height:1;
                text-shadow:0 0 20px {color};
            ">{num}</p>
            <p style="
                font-family:'Orbitron',sans-serif;
                font-size:0.72rem;
                font-weight:700;
                color:{color};
                margin:0 0 6px 0;
                letter-spacing:0.1em;
                text-transform:uppercase;
                text-shadow:0 0 10px {color}66;
            ">{title}</p>
            <p style="color:#4A3580; font-family:'Rajdhani',sans-serif; font-size:0.9rem; line-height:1.55; margin:0;">{desc}</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE: SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📷 Image":
    uploaded = st.file_uploader(
        "Drop a vehicle image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        model = load_model()
        img_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        probs, classes = predict(model, img_bgr)
        top_idx = int(np.argmax(probs))
        top_cls = classes[top_idx]
        top_conf = float(probs[top_idx])

        # Top-3
        top3_idx = np.argsort(probs)[::-1][:3]

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<p class="section-header">Input Image</p>', unsafe_allow_html=True)
            st.image(img_pil, use_container_width=True)

            # Top-3 accuracy strip
            st.markdown('<p class="section-header" style="margin-top:16px;">Top-3 Predictions</p>', unsafe_allow_html=True)
            t3cols = st.columns(3)
            for rank, (tc, ti) in enumerate(zip(t3cols, top3_idx)):
                cls_name = classes[ti]
                conf_val = float(probs[ti])
                color = CLASS_COLORS.get(cls_name, '#4A3580')
                icon = CLASS_ICONS.get(cls_name, '🚘')
                medal = ["🥇", "🥈", "🥉"][rank]
                tc.markdown(f"""
                <div class="batch-card">
                    <div class="batch-card-icon">{medal} {icon}</div>
                    <div class="batch-card-label" style="color:{color}">{cls_name}</div>
                    <div class="batch-card-conf">{conf_val*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-header">Prediction</p>', unsafe_allow_html=True)

            if top_conf >= conf_threshold:
                color = CLASS_COLORS.get(top_cls, '#ffffff')
                icon = CLASS_ICONS.get(top_cls, '🚘')
                st.markdown(f"""
                <div class="pred-card">
                    <p style="font-size:3rem;margin:0">{icon}</p>
                    <p class="pred-label" style="color:{color}">{top_cls}</p>
                    <p class="pred-conf">{top_conf*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Top prediction ({top_conf*100:.1f}%) is below threshold ({conf_threshold*100:.0f}%)")

            st.markdown(render_prob_bars(probs, classes), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color:#2A0A5E;">
            <p style="font-size:3rem">📷</p>
            <p style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD AN IMAGE TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🖼️ Batch":
    st.markdown("### Batch Prediction")
    st.markdown("<p style='color:#4A3580;'>Upload multiple vehicle images and get predictions for all at once.</p>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        model = load_model()
        results_data = []

        progress_bar = st.progress(0)
        status_txt = st.empty()

        for i, f in enumerate(uploaded_files):
            img_pil = Image.open(f).convert("RGB")
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            probs, classes = predict(model, img_bgr)
            top_idx = int(np.argmax(probs))
            top3_idx = np.argsort(probs)[::-1][:3]
            results_data.append({
                "filename": f.name,
                "image": img_pil,
                "probs": probs,
                "classes": classes,
                "top_idx": top_idx,
                "top_cls": classes[top_idx],
                "top_conf": float(probs[top_idx]),
                "top3": [(classes[j], float(probs[j])) for j in top3_idx],
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_txt.text(f"Processing {i+1}/{len(uploaded_files)}: {f.name}")

        progress_bar.empty()
        status_txt.empty()

        # ── Summary stats ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">Batch Summary</p>', unsafe_allow_html=True)

        total = len(results_data)
        above_thresh = sum(1 for r in results_data if r["top_conf"] >= conf_threshold)
        avg_conf = np.mean([r["top_conf"] for r in results_data]) * 100
        class_counts = {c: sum(1 for r in results_data if r["top_cls"] == c) for c in CLASSES}
        dominant_cls = max(class_counts, key=class_counts.get)

        sc = st.columns(4)
        stats = [
            (str(total), "Total Images"),
            (f"{above_thresh}", "Above Threshold"),
            (f"{avg_conf:.1f}%", "Avg Confidence"),
            (f"{CLASS_ICONS[dominant_cls]} {dominant_cls}", "Most Common"),
        ]
        for col, (val, lbl) in zip(sc, stats):
            col.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#C8B8FF">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # ── Class distribution pie ─────────────────────────────────────────────
        st.markdown("---")
        dist_col, export_col = st.columns([1, 1], gap="large")
        with dist_col:
            st.markdown('<p class="section-header">Class Distribution</p>', unsafe_allow_html=True)
            labels = [c for c in CLASSES if class_counts[c] > 0]
            sizes  = [class_counts[c] for c in labels]
            colors = [CLASS_COLORS[c] for c in labels]

            fig, ax = plt.subplots(figsize=(4, 3.2))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.0f%%', startangle=140,
                textprops={'color': PLT_TEXT, 'fontsize': 9},
                wedgeprops={'edgecolor': PLT_BG, 'linewidth': 2}
            )
            for at in autotexts:
                at.set_color(PLT_BG)
                at.set_fontweight('bold')
                at.set_fontsize(9)
            apply_dark_theme(fig, [ax])
            ax.set_facecolor(PLT_BG)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with export_col:
            st.markdown('<p class="section-header">Export Results</p>', unsafe_allow_html=True)
            df_export = pd.DataFrame([{
                "Filename":    r["filename"],
                "Prediction":  r["top_cls"],
                "Confidence":  f"{r['top_conf']*100:.2f}%",
                "Top2":        r["top3"][1][0] if len(r["top3"]) > 1 else "",
                "Top2_Conf":   f"{r['top3'][1][1]*100:.2f}%" if len(r["top3"]) > 1 else "",
                "Top3":        r["top3"][2][0] if len(r["top3"]) > 2 else "",
                "Top3_Conf":   f"{r['top3'][2][1]*100:.2f}%" if len(r["top3"]) > 2 else "",
            } for r in results_data])
            st.dataframe(df_export, use_container_width=True, height=220)
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV",
                csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── Grid of results ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">All Results</p>', unsafe_allow_html=True)

        COLS_PER_ROW = 4
        for row_start in range(0, len(results_data), COLS_PER_ROW):
            row_items = results_data[row_start: row_start + COLS_PER_ROW]
            cols = st.columns(COLS_PER_ROW)
            for col, r in zip(cols, row_items):
                cls = r["top_cls"]
                conf = r["top_conf"]
                color = CLASS_COLORS.get(cls, '#4A3580')
                icon = CLASS_ICONS.get(cls, '🚘')
                dim = (180, 180)
                thumb = r["image"].copy()
                thumb.thumbnail(dim, Image.LANCZOS)
                col.image(thumb, use_container_width=True)
                badge = "✅" if conf >= conf_threshold else "⚠️"
                col.markdown(f"""
                <div class="batch-card">
                    <div class="batch-card-label" style="color:{color}">{badge} {icon} {cls}</div>
                    <div class="batch-card-conf">{conf*100:.1f}%</div>
                    <div class="batch-card-filename">{r['filename']}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color:#2A0A5E;">
            <p style="font-size:3rem">🖼️</p>
            <p style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD MULTIPLE IMAGES TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# MODE: TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Training Curves":
    st.markdown("### Training Curves")
    st.markdown(
        "<p style='color:#4A3580;'>Upload your <code>results.csv</code> exported by YOLOv8 training to visualise training dynamics.</p>",
        unsafe_allow_html=True
    )

    csv_file = st.file_uploader(
        "Upload results.csv",
        type=["csv"],
        label_visibility="collapsed"
    )

    # ── Column-name normaliser ─────────────────────────────────────────────────
    def norm(col):
        """Lowercase, strip spaces → canonical key."""
        return col.strip().lower().replace(" ", "").replace("/", "_").replace("(", "").replace(")", "")

    # ── Known column mappings (ultralytics results.csv) ───────────────────────
    COL_MAP = {
        # losses
        "train_box_loss":  ["train/box_loss"],
        "train_cls_loss":  ["train/cls_loss"],
        "val_box_loss":    ["val/box_loss"],
        "val_cls_loss":    ["val/cls_loss"],
        # classification accuracy
        "metrics_top1":    ["metrics/accuracy_top1"],
        "metrics_top5":    ["metrics/accuracy_top5"],
        # learning rate
        "lr_pg0":          ["lr/pg0"],
    }

    def find_col(df, candidates):
        norm_map = {norm(c): c for c in df.columns}
        for cand in candidates:
            key = norm(cand)
            if key in norm_map:
                return norm_map[key]
        return None

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns]

            # epoch column
            epoch_col = None
            for c in df.columns:
                if "epoch" in norm(c):
                    epoch_col = c
                    break
            epochs = df[epoch_col].values if epoch_col else np.arange(len(df))

            # ── Detect available curves ────────────────────────────────────────
            detected = {}
            for key, cands in COL_MAP.items():
                col = find_col(df, cands)
                if col:
                    detected[key] = df[col].values

            if not detected:
                st.warning("⚠️ No recognised columns found. Make sure this is a YOLOv8 `results.csv`.")
                st.markdown("**Columns found:**")
                st.code(", ".join(df.columns))
            else:
                # ── Plot 1: Loss curves ────────────────────────────────────────
                loss_keys = [k for k in detected if "loss" in k]
                acc_keys  = [k for k in detected if "top" in k or "accuracy" in k]

                plot_cols = st.columns(2)

                # Loss
                with plot_cols[0]:
                    st.markdown('<p class="section-header">Loss Curves</p>', unsafe_allow_html=True)
                    if loss_keys:
                        fig, ax = plt.subplots(figsize=(5, 3.2))
                        for k in loss_keys:
                            label = k.replace("train_", "Train ").replace("val_", "Val ").replace("_loss", " Loss").replace("_", " ").title()
                            style = "--" if "val" in k else "-"
                            color = CURVE_COLORS["val"] if "val" in k else CURVE_COLORS["train"]
                            ax.plot(epochs, detected[k], linestyle=style, color=color, linewidth=1.8, label=label, alpha=0.9)
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        ax.set_title("Training & Validation Loss")
                        ax.legend(fontsize=8, labelcolor=PLT_TEXT, facecolor=PLT_SURFACE, edgecolor=PLT_BORDER)
                        apply_dark_theme(fig, [ax])
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("No loss columns detected.")

                # Accuracy
                with plot_cols[1]:
                    st.markdown('<p class="section-header">Accuracy Curves</p>', unsafe_allow_html=True)
                    if acc_keys:
                        fig, ax = plt.subplots(figsize=(5, 3.2))
                        for k in acc_keys:
                            label = "Top-1 Accuracy" if "top1" in k else "Top-5 Accuracy"
                            color = CURVE_COLORS["top1"] if "top1" in k else CURVE_COLORS["top3"]
                            ax.plot(epochs, detected[k], linewidth=1.8, color=color, label=label)
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Top-1 & Top-5 Accuracy")
                        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                        ax.legend(fontsize=8, labelcolor=PLT_TEXT, facecolor=PLT_SURFACE, edgecolor=PLT_BORDER)
                        apply_dark_theme(fig, [ax])
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("No accuracy columns detected.")

                # ── Plot 2: LR ─────────────────────────────────────────────────
                lr_keys = [k for k in detected if "lr" in k]
                if lr_keys:
                    st.markdown('<p class="section-header">Learning Rate Schedule</p>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(8, 2.2))
                    ax.plot(epochs, detected[lr_keys[0]], linewidth=1.6, color="#BF00FF")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("LR")
                    ax.set_title("Learning Rate (pg0)")
                    apply_dark_theme(fig, [ax])
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # ── Summary table ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<p class="section-header">Training Summary</p>', unsafe_allow_html=True)

                summary = {}
                if "metrics_top1" in detected:
                    best_e = int(np.argmax(detected["metrics_top1"]))
                    summary["Best Top-1 Acc"] = f"{detected['metrics_top1'][best_e]*100:.2f}% (epoch {int(epochs[best_e])})"
                if "metrics_top5" in detected:
                    best_e5 = int(np.argmax(detected["metrics_top5"]))
                    summary["Best Top-5 Acc"] = f"{detected['metrics_top5'][best_e5]*100:.2f}% (epoch {int(epochs[best_e5])})"
                for lk in loss_keys:
                    label = lk.replace("_loss", "").replace("_", " ").title() + " (final)"
                    summary[label] = f"{detected[lk][-1]:.4f}"
                summary["Total Epochs"] = str(len(epochs))

                s_cols = st.columns(min(len(summary), 4))
                for col, (k, v) in zip(s_cols * 10, summary.items()):
                    col.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" style="color:#C8B8FF;font-size:1.1rem;">{v}</div>
                        <div class="stat-label">{k}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Raw data ───────────────────────────────────────────────────
                with st.expander("Raw CSV data"):
                    st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to parse CSV: {e}")

    else:
        # ── Demo / instructions ────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#0D0720;border:1px solid #2A0A5E;border-radius:12px;padding:24px;margin-top:16px;">
            <p style="font-family:'Share Tech Mono',monospace;color:#4A3580;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">Expected Columns</p>
            <ul style="color:#C8B8FF;line-height:2;">
                <li><code>epoch</code></li>
                <li><code>train/box_loss</code>, <code>train/cls_loss</code></li>
                <li><code>val/box_loss</code>, <code>val/cls_loss</code></li>
                <li><code>metrics/accuracy_top1</code>, <code>metrics/accuracy_top5</code></li>
                <li><code>lr/pg0</code></li>
            </ul>
            <p style="color:#4A3580;font-size:0.85rem;">
                YOLOv8 saves this file automatically to <code>runs/classify/train/results.csv</code>
                after training completes.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🗂️ Dataset":
    st.markdown("### Dataset Overview")
    st.markdown("<p style='color:#4A3580;'>Class distribution across the training split (~4,081 images).</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Hardcoded train-split counts from the dataset ─────────────────────────
    DATASET_COUNTS = {
        'Bus':        1358,
        'Car':        700,
        'Motorcycle': 845,
        'Truck':      1178,
    }
    total_train = sum(DATASET_COUNTS.values())

    # ── Summary stat boxes ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Train Split Stats</p>', unsafe_allow_html=True)
    sc = st.columns(5)
    summary_stats = [
        (str(total_train), "Total Images"),
        (str(DATASET_COUNTS['Bus']),        "Bus"),
        (str(DATASET_COUNTS['Car']),        "Car"),
        (str(DATASET_COUNTS['Motorcycle']), "Motorcycle"),
        (str(DATASET_COUNTS['Truck']),      "Truck"),
    ]
    stat_colors = ["#C8B8FF", CLASS_COLORS['Bus'], CLASS_COLORS['Car'], CLASS_COLORS['Motorcycle'], CLASS_COLORS['Truck']]
    for col, (val, lbl), color in zip(sc, summary_stats, stat_colors):
        col.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:{color}">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Side-by-side: bar chart + pie chart ───────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2], gap="large")

    classes  = list(DATASET_COUNTS.keys())
    counts   = list(DATASET_COUNTS.values())
    colors   = [CLASS_COLORS[c] for c in classes]

    with chart_col1:
        st.markdown('<p class="section-header">Images per Class</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(classes, counts, color=colors, edgecolor=PLT_BG, linewidth=1.5, width=0.55)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 18,
                str(count),
                ha='center', va='bottom',
                color=PLT_TEXT, fontsize=10,
                fontfamily='monospace', fontweight='bold'
            )
        ax.set_ylabel("Count")
        ax.set_title("Dataset Class Distribution (Train Split)", pad=14)
        ax.set_ylim(0, max(counts) * 1.15)
        apply_dark_theme(fig, [ax])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_col2:
        st.markdown('<p class="section-header">Class Proportion</p>', unsafe_allow_html=True)
        proportions = [c / total_train for c in counts]
        fig2, ax2 = plt.subplots(figsize=(4.2, 4))
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=classes,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': PLT_TEXT, 'fontsize': 9},
            wedgeprops={'edgecolor': PLT_BG, 'linewidth': 2},
            pctdistance=0.65,
        )
        for at in autotexts:
            at.set_color(PLT_BG)
            at.set_fontweight('bold')
            at.set_fontsize(9)
        ax2.set_title("Class Proportion", color=PLT_TEXT, pad=10)
        apply_dark_theme(fig2, [ax2])
        ax2.set_facecolor(PLT_BG)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Dataset split breakdown ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Dataset Split</p>', unsafe_allow_html=True)

    TOTAL_DATASET = 8863
    split_data = {
        "Train": (0.70, "#00F5FF"),
        "Val":   (0.15, "#00FF9F"),
        "Test":  (0.15, "#FF2079"),
    }
    sp_cols = st.columns(3)
    for col, (split_name, (ratio, color)) in zip(sp_cols, split_data.items()):
        n = int(TOTAL_DATASET * ratio)
        col.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:{color}">{n:,}</div>
            <div class="stat-label">{split_name} ({int(ratio*100)}%)</div>
        </div>""", unsafe_allow_html=True)

    # Stacked split bar
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    bar_html = "<div style='display:flex;height:14px;border-radius:7px;overflow:hidden;gap:2px;margin-top:8px;'>"
    for split_name, (ratio, color) in split_data.items():
        bar_html += f"<div style='flex:{ratio};background:{color};border-radius:7px;' title='{split_name}'></div>"
    bar_html += "</div>"
    st.markdown(bar_html, unsafe_allow_html=True)
    # legend
    legend_html = "<div style='display:flex;gap:20px;margin-top:8px;'>"
    for split_name, (ratio, color) in split_data.items():
        legend_html += f"<span style='font-size:0.75rem;color:{color};font-family:Share Tech Mono,monospace;'>■ {split_name} {int(ratio*100)}%</span>"
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "ℹ️ About":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Project Summary")
        st.markdown("""
        This system uses **YOLOv8s-cls** fine-tuned on a vehicle dataset to classify
        images into four categories in real time.

        **Pipeline:**
        1. Input image/frame → grayscale preprocessing
        2. YOLOv8 classification inference (224×224)
        3. Softmax probabilities → top prediction
        """)

    with col2:
        st.markdown("### Model Details")
        data = {
            "Architecture": "YOLOv8s-cls",
            "Input size": "224 × 224 px",
            "Classes": "Bus, Car, Motorcycle, Truck",
            "Dataset": "~8,863 images",
            "Split": "70% / 15% / 15%",
            "Preprocessing": "Grayscale normalisation",
        }
        for k, v in data.items():
            st.markdown(f"**{k}:** {v}")

    st.markdown("### Technologies")
    cols = st.columns(4)
    techs = ["Python", "YOLOv8", "OpenCV", "Streamlit"]
    for col, t in zip(cols, techs):
        col.markdown(f'<div class="tech-chip">{t}</div>', unsafe_allow_html=True)
