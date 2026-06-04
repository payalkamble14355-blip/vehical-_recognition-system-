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
    'Bus':        '#ffb6c1',   # light pink
    'Car':        '#fffacd',   # lemon chiffon
    'Motorcycle': '#98fb98',   # pale green
    'Truck':      '#87cefa',   # light sky blue
}
CLASS_ICONS = {
    'Bus': '🚌',
    'Car': '🚗',
    'Motorcycle': '🏍️',
    'Truck': '🚛',
}

# ── Plot theme helper ──────────────────────────────────────────────────────────
PLT_BG       = "#0a0e1a"
PLT_SURFACE  = "#0d1526"
PLT_BORDER   = "#00e5ff"
PLT_TEXT     = "#00e5ff"
PLT_SUBTEXT  = "#1a4a5a"

CURVE_COLORS = {
    "train": "#00e5ff",
    "val":   "#7b2fff",
    "top1":  "#00e5ff",
    "top3":  "#7b2fff",
    "bus":   "#00e5ff",
    "car":   "#7b2fff",
    "motorcycle": "#00ffaa",
    "truck": "#ff6ec7",
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
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        color: #a8d8e8;
        background-color: #0a0e1a;
    }
    h1, h2, h3 { font-family: 'Orbitron', monospace; color: #00e5ff; }

    .main  { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #060a14 !important;
        border-right: 1px solid #00e5ff22 !important;
    }
    [data-testid="stSidebar"] * { color: #a8d8e8 !important; }
    [data-testid="stSidebar"] hr { border-color: #00e5ff22 !important; }
    [data-testid="stSidebar"] label {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.06em !important;
    }

    /* Slider */
    [data-testid="stSlider"] > div > div > div { background: #00e5ff !important; }

    hr { border-color: #00e5ff22 !important; }

    /* ── Prediction card — glowing cyan box ── */
    .pred-card {
        background: linear-gradient(145deg, #0d1a2e 0%, #061020 100%);
        border: 2px solid #00e5ff;
        border-radius: 16px;
        padding: 36px 28px;
        margin: 12px 0;
        text-align: center;
        box-shadow:
            0 0 20px #00e5ff55,
            0 0 60px #00e5ff22,
            inset 0 0 30px #00e5ff08;
        animation: fadeSlideIn 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    .pred-card::before {
        content: '';
        position: absolute;
        top: 0; left: 10%; right: 10%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        box-shadow: 0 0 10px #00e5ff;
    }
    .pred-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 10%; right: 10%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00e5ff55, transparent);
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .pred-label {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 0.12em;
        text-shadow: 0 0 20px currentColor, 0 0 40px currentColor;
    }
    .pred-conf {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #00e5ff;
        margin-top: 10px;
        letter-spacing: 0.1em;
        text-shadow: 0 0 10px #00e5ff88;
    }

    /* ── Probability bars ── */
    .prob-bar-container {
        background: #060a14;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #00e5ff22;
        margin: 10px 0;
        box-shadow: 0 0 20px #00e5ff08;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 10px 0;
        gap: 12px;
    }
    .prob-label {
        width: 120px;
        font-size: 0.88rem;
        color: #a8d8e8;
        font-weight: 500;
        letter-spacing: 0.06em;
        font-family: 'Rajdhani', sans-serif;
    }
    .prob-track {
        flex: 1;
        height: 8px;
        background: #0d1526;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #00e5ff11;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.7s ease;
        box-shadow: 0 0 8px currentColor;
    }
    .prob-val {
        width: 50px;
        text-align: right;
        font-size: 0.85rem;
        color: #00e5ff;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
    }

    /* ── Section header ── */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #1a6a7a;
        margin-bottom: 14px;
        border-bottom: 1px solid #00e5ff22;
        padding-bottom: 6px;
    }

    /* ── Batch result card ── */
    .batch-card {
        background: #060a14;
        border: 1px solid #00e5ff22;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .batch-card:hover {
        transform: translateY(-4px);
        border-color: #00e5ff;
        box-shadow: 0 0 20px #00e5ff33;
    }
    .batch-card-icon { font-size: 2rem; margin-bottom: 6px; }
    .batch-card-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .batch-card-conf {
        font-size: 0.75rem;
        color: #1a6a7a;
        margin-top: 3px;
        letter-spacing: 0.06em;
    }
    .batch-card-filename {
        font-size: 0.65rem;
        color: #0d2a30;
        margin-top: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Summary stat box ── */
    .stat-box {
        background: #060a14;
        border: 1px solid #00e5ff22;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.2s;
    }
    .stat-box:hover { box-shadow: 0 0 20px #00e5ff22; border-color: #00e5ff55; }
    .stat-box::after {
        content: '';
        position: absolute;
        bottom: 0; left: 20%; right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00e5ff44, transparent);
    }
    .stat-number {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: 0.04em;
    }
    .stat-label {
        font-size: 0.65rem;
        color: #1a6a7a;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-top: 6px;
        font-family: 'Rajdhani', sans-serif;
    }

    /* ── Upload area ── */
    div[data-testid="stFileUploader"] {
        background: #060a14;
        border: 1px dashed #00e5ff33;
        border-radius: 12px;
        padding: 10px;
    }

    /* ── Dataset card ── */
    .ds-card {
        background: #060a14;
        border: 1px solid #00e5ff22;
        border-radius: 10px;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .ds-card:hover { transform: translateY(-3px); border-color: #00e5ff; box-shadow: 0 0 16px #00e5ff22; }
    .ds-card-label {
        padding: 8px 10px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #a8d8e8;
    }
    .ds-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        letter-spacing: 0.06em;
    }

    /* ── Tech chip ── */
    .tech-chip {
        background: #060a14;
        border: 1px solid #00e5ff22;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 0.72rem;
        color: #a8d8e8;
        letter-spacing: 0.1em;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .tech-chip:hover { border-color: #00e5ff; box-shadow: 0 0 16px #00e5ff22; }

    /* ── Buttons ── */
    .stDownloadButton button, .stButton button {
        background: transparent !important;
        border: 1px solid #00e5ff !important;
        color: #00e5ff !important;
        border-radius: 6px !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        transition: background 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 0 8px #00e5ff22 !important;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background: #00e5ff12 !important;
        box-shadow: 0 0 20px #00e5ff44 !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #00e5ff22 !important;
        border-radius: 10px !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background: #060a14 !important;
        border: 1px solid #00e5ff44 !important;
        border-radius: 10px !important;
        color: #a8d8e8 !important;
    }

    /* ── Scanline overlay for cyberpunk feel ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,229,255,0.015) 2px,
            rgba(0,229,255,0.015) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 14px 0 10px 0;'>
        <p style='font-size:2rem; margin:0; filter: drop-shadow(0 0 8px #00e5ff);'>🚗</p>
        <p style='font-family:"Orbitron",monospace; font-size:1rem; font-weight:700;
                  color:#00e5ff; letter-spacing:0.14em; margin:8px 0 0 0;
                  text-shadow: 0 0 12px #00e5ff88;'>VEHICLE AI</p>
        <p style='font-size:0.55rem; letter-spacing:0.24em; color:#1a6a7a;
                  text-transform:uppercase; margin:4px 0 0 0;'>Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio(
        "Mode",
        ["🏠 Home", "📷 Image", "🖼️ Batch", "📈 Training Curves", "🗂️ Dataset", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<p style='font-size:0.55rem;letter-spacing:0.22em;color:#1a6a7a;text-transform:uppercase;margin-bottom:6px;font-family:Orbitron,monospace;'>Model</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a8d8e8;font-size:0.85rem;margin:0;font-family:Rajdhani,sans-serif;'>YOLOv8s-cls</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.55rem;letter-spacing:0.22em;color:#1a6a7a;text-transform:uppercase;margin:12px 0 6px 0;font-family:Orbitron,monospace;'>Classes</p>", unsafe_allow_html=True)
    for cls in CLASSES:
        st.markdown(f"<p style='color:#a8d8e8;font-size:0.85rem;margin:3px 0;font-family:Rajdhani,sans-serif;'>{CLASS_ICONS[cls]} {cls}</p>", unsafe_allow_html=True)
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
        color = CLASS_COLORS.get(cls, '#7b84a8')
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
st.markdown("""
<div style='margin-bottom:4px;'>
    <h1 style='font-family:"Orbitron",monospace; font-size:2rem; font-weight:900;
               color:#00e5ff; letter-spacing:0.1em; margin:0;
               text-shadow: 0 0 20px #00e5ff66, 0 0 40px #00e5ff33;'>
        VEHICLE RECOGNITION SYSTEM
    </h1>
</div>
<p style='color:#1a6a7a; font-size:0.6rem; letter-spacing:0.26em; text-transform:uppercase;
          margin:0 0 16px 0; font-family:"Orbitron",monospace;'>
    YOLOV8 &nbsp;·&nbsp; BUS &nbsp;·&nbsp; CAR &nbsp;·&nbsp; MOTORCYCLE &nbsp;·&nbsp; TRUCK
</p>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#00e5ff22; margin:0 0 24px 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🏠 Home":
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(145deg, #0d1526 0%, #060a14 60%, #0a1020 100%);
        border: 1px solid #00e5ff33;
        border-top: 2px solid #00e5ff;
        border-radius: 16px;
        padding: 56px 48px 48px;
        text-align: center;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px #00e5ff0a, inset 0 0 80px #00000040;
    ">
        <div style="position:absolute;top:-80px;left:-80px;width:280px;height:280px;
            background:radial-gradient(circle,#00e5ff0a 0%,transparent 70%);"></div>
        <div style="position:absolute;bottom:-80px;right:-80px;width:280px;height:280px;
            background:radial-gradient(circle,#7b2fff0a 0%,transparent 70%);"></div>
        <p style="font-size:3.5rem;margin:0 0 16px 0;line-height:1;
                  filter:drop-shadow(0 0 12px #00e5ff);">🚗</p>
        <h1 style="font-family:'Orbitron',monospace;font-size:2.4rem;font-weight:900;
                   color:#00e5ff;margin:0 0 8px 0;letter-spacing:0.1em;
                   text-shadow:0 0 30px #00e5ff88, 0 0 60px #00e5ff33;">
            VEHICLE RECOGNITION
        </h1>
        <p style="color:#1a6a7a;font-family:'Orbitron',monospace;font-size:0.58rem;
                  letter-spacing:0.3em;text-transform:uppercase;margin:0 0 28px 0;">
            YOLOV8S-CLS &nbsp;·&nbsp; BUS &nbsp;·&nbsp; CAR &nbsp;·&nbsp; MOTORCYCLE &nbsp;·&nbsp; TRUCK
        </p>
        <div style="width:80px;height:1px;background:linear-gradient(90deg,transparent,#00e5ff,transparent);
                    margin:0 auto 24px;box-shadow:0 0 8px #00e5ff;"></div>
        <p style="color:#4a9aaa;font-size:0.95rem;max-width:520px;margin:0 auto;line-height:1.8;
                  font-family:'Rajdhani',sans-serif;letter-spacing:0.04em;">
            A deep learning system that classifies vehicle images into four categories
            in real time using a fine-tuned YOLOv8s classification model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-stats row ───────────────────────────────────────────────────────
    qs = st.columns(4)
    quick_stats = [
        ("8,863",    "Total Images",   "#00e5ff"),
        ("4",        "Classes",        "#7b2fff"),
        ("224px",    "Input Size",     "#00ffaa"),
        ("70/15/15", "Train/Val/Test", "#ff6ec7"),
    ]
    for col, (val, lbl, color) in zip(qs, quick_stats):
        col.markdown(f"""
        <div class="stat-box" style="border-color:{color}33;">
            <div class="stat-number" style="color:{color}; text-shadow:0 0 12px {color}88;">{val}</div>
            <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">What you can do</p>', unsafe_allow_html=True)
    fc = st.columns(4)
    features = [
        ("📷", "Image",           "Upload a single image and get an instant classification with confidence scores and probability bars."),
        ("🖼️", "Batch",           "Predict on multiple images at once. Export results to CSV and see class distribution at a glance."),
        ("📈", "Training Curves", "Upload your YOLOv8 results.csv to visualise loss, accuracy, and learning-rate curves over epochs."),
        ("🗂️", "Dataset",         "Explore the dataset composition — class counts, train/val/test splits, and proportion charts."),
    ]
    for col, (icon, title, desc) in zip(fc, features):
        col.markdown(f"""
        <div style="
            background: #060a14;
            border: 1px solid #00e5ff22;
            border-top: 2px solid #00e5ff;
            border-radius: 12px;
            padding: 22px 18px;
            height: 100%;
            transition: box-shadow 0.2s, border-color 0.2s;
            box-shadow: 0 0 0 transparent;
        ">
            <p style="font-size:1.6rem; margin:0 0 12px 0; filter:drop-shadow(0 0 6px #00e5ff66);">{icon}</p>
            <p style="font-family:'Orbitron',monospace;font-size:0.72rem;font-weight:700;
                      color:#00e5ff;margin:0 0 8px 0;letter-spacing:0.1em;
                      text-shadow:0 0 10px #00e5ff66;">{title}</p>
            <p style="color:#2a7a8a; font-size:0.82rem; line-height:1.7; margin:0;
                      font-family:'Rajdhani',sans-serif;">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

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
        <div style="background:#060a14;border:1px solid #00e5ff22;border-radius:12px;
                    padding:22px 16px;text-align:center;">
            <p style="font-size:2.2rem;margin:0 0 10px 0;
                      filter:drop-shadow(0 0 8px {color}88);">{icon}</p>
            <p style="font-family:'Orbitron',monospace;font-size:0.78rem;font-weight:700;
                      color:{color};margin:0 0 8px 0;letter-spacing:0.1em;
                      text-shadow:0 0 10px {color}66;">{cls}</p>
            <div style="width:30px;height:1px;background:#00e5ff22;margin:0 auto 10px;"></div>
            <p style="color:#2a7a8a;font-size:0.75rem;line-height:1.6;margin:0;
                      font-family:'Rajdhani',sans-serif;">{class_descs[cls]}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">How it works</p>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload",     "Drop an image or batch of images into the app."),
        ("02", "Preprocess", "Converted to grayscale and resized to 224×224."),
        ("03", "Infer",      "YOLOv8s-cls runs a forward pass and outputs class logits."),
        ("04", "Result",     "Softmax probabilities are ranked and displayed with labels."),
    ]
    pipe_cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(pipe_cols, steps):
        col.markdown(f"""
        <div style="background:#060a14;border:1px solid #00e5ff22;border-radius:12px;
                    padding:20px 14px;text-align:center;position:relative;">
            <p style="font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;
                      color:#00e5ff0f;margin:0 0 6px 0;line-height:1;">{num}</p>
            <p style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;
                      color:#00e5ff;margin:0 0 8px 0;letter-spacing:0.1em;
                      text-shadow:0 0 10px #00e5ff66;">{title}</p>
            <p style="color:#2a7a8a;font-size:0.75rem;line-height:1.6;margin:0;
                      font-family:'Rajdhani',sans-serif;">{desc}</p>
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
                color = CLASS_COLORS.get(cls_name, '#7b84a8')
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
        <div style="text-align:center; padding: 60px 20px; color:#3a4060;">
            <p style="font-size:3rem">📷</p>
            <p style="font-family:'Orbitron',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD AN IMAGE TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🖼️ Batch":
    st.markdown("### Batch Prediction")
    st.markdown("<p style='color:#1a6a7a;'>Upload multiple vehicle images and get predictions for all at once.</p>", unsafe_allow_html=True)

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
                <div class="stat-number" style="color:#00e5ff">{val}</div>
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
                color = CLASS_COLORS.get(cls, '#7b84a8')
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
        <div style="text-align:center; padding: 60px 20px; color:#3a4060;">
            <p style="font-size:3rem">🖼️</p>
            <p style="font-family:'Orbitron',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD MULTIPLE IMAGES TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# MODE: TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Training Curves":
    st.markdown("### Training Curves")
    st.markdown(
        "<p style='color:#1a6a7a;'>Upload your <code>results.csv</code> exported by YOLOv8 training to visualise training dynamics.</p>",
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
                    ax.plot(epochs, detected[lr_keys[0]], linewidth=1.6, color="#7b2fff")
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
                        <div class="stat-number" style="color:#00e5ff;font-size:1.1rem;">{v}</div>
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
        <div style="background:#1a1d2e;border:1px solid #2d3561;border-radius:12px;padding:24px;margin-top:16px;">
            <p style="font-family:'Orbitron',monospace;color:#1a6a7a;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">Expected Columns</p>
            <ul style="color:#00e5ff;line-height:2;">
                <li><code>epoch</code></li>
                <li><code>train/box_loss</code>, <code>train/cls_loss</code></li>
                <li><code>val/box_loss</code>, <code>val/cls_loss</code></li>
                <li><code>metrics/accuracy_top1</code>, <code>metrics/accuracy_top5</code></li>
                <li><code>lr/pg0</code></li>
            </ul>
            <p style="color:#1a6a7a;font-size:0.85rem;">
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
    st.markdown("<p style='color:#1a6a7a;'>Class distribution across the training split (~4,081 images).</p>", unsafe_allow_html=True)
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
    stat_colors = ["#00e5ff", CLASS_COLORS['Bus'], CLASS_COLORS['Car'], CLASS_COLORS['Motorcycle'], CLASS_COLORS['Truck']]
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
        "Train": (0.70, "#00e5ff"),
        "Val":   (0.15, "#7b2fff"),
        "Test":  (0.15, "#ff6ec7"),
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
        legend_html += f"<span style='font-size:0.75rem;color:{color};font-family:Orbitron,monospace;'>■ {split_name} {int(ratio*100)}%</span>"
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
