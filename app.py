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
    'Bus':        '#C9A84C',   # gold
    'Car':        '#E2C07A',   # light gold
    'Motorcycle': '#8B6914',   # deep gold
    'Truck':      '#D4AF37',   # metallic gold
}
CLASS_ICONS = {
    'Bus': '🚌',
    'Car': '🚗',
    'Motorcycle': '🏍️',
    'Truck': '🚛',
}

# ── Plot theme helper ──────────────────────────────────────────────────────────
PLT_BG       = "#0A0A0A"
PLT_SURFACE  = "#111111"
PLT_BORDER   = "#2A2200"
PLT_TEXT     = "#E2C07A"
PLT_SUBTEXT  = "#5C4A1A"

CURVE_COLORS = {
    "train": "#C9A84C",
    "val":   "#8B6914",
    "top1":  "#E2C07A",
    "top3":  "#D4AF37",
    "bus":   "#C9A84C",
    "car":   "#E2C07A",
    "motorcycle": "#8B6914",
    "truck": "#D4AF37",
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
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=EB+Garamond:wght@400;500;600&family=Cinzel:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'EB Garamond', Georgia, serif; }
    h1, h2, h3 { font-family: 'Cinzel', serif; letter-spacing: 0.08em; }

    .main { background-color: #0A0A0A; }
    .stApp { background-color: #0A0A0A; }

    /* ── Prediction card ── */
    .pred-card {
        background: linear-gradient(160deg, #111111 0%, #0A0A0A 60%, #150F00 100%);
        border: 1px solid #2A2200;
        border-top: 2px solid #C9A84C;
        border-radius: 4px;
        padding: 32px 24px;
        margin: 12px 0;
        text-align: center;
        animation: fadeSlideIn 0.5s ease;
        box-shadow: 0 8px 40px rgba(201,168,76,0.08), inset 0 1px 0 rgba(201,168,76,0.1);
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .pred-label {
        font-family: 'Cinzel', serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .pred-conf {
        font-family: 'EB Garamond', serif;
        font-size: 1.05rem;
        color: #7A6030;
        margin-top: 6px;
        font-style: italic;
        letter-spacing: 0.04em;
    }

    /* ── Probability bars ── */
    .prob-bar-container {
        background: #0D0D0D;
        border-radius: 4px;
        padding: 20px 24px;
        border: 1px solid #2A2200;
        margin: 8px 0;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 10px 0;
        gap: 12px;
    }
    .prob-label {
        width: 120px;
        font-family: 'EB Garamond', serif;
        font-size: 1rem;
        color: #C9A84C;
        font-style: italic;
    }
    .prob-track {
        flex: 1;
        height: 3px;
        background: #1A1400;
        border-radius: 0;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 0;
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        background: linear-gradient(90deg, #8B6914, #C9A84C) !important;
    }
    .prob-val {
        width: 52px;
        text-align: right;
        font-size: 0.9rem;
        color: #7A6030;
        font-family: 'EB Garamond', serif;
    }

    /* ── Section header ── */
    .section-header {
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #7A6030;
        margin-bottom: 14px;
        border-bottom: 1px solid #1A1400;
        padding-bottom: 8px;
    }

    /* ── Batch result card ── */
    .batch-card {
        background: #0D0D0D;
        border: 1px solid #1A1400;
        border-top: 2px solid #C9A84C44;
        border-radius: 2px;
        padding: 14px 12px;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .batch-card:hover {
        transform: translateY(-4px);
        border-color: #C9A84C;
        box-shadow: 0 12px 30px rgba(201,168,76,0.1);
    }
    .batch-card-icon { font-size: 1.8rem; margin-bottom: 6px; }
    .batch-card-label {
        font-family: 'Cinzel', serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .batch-card-conf {
        font-family: 'EB Garamond', serif;
        font-size: 0.9rem;
        color: #7A6030;
        margin-top: 2px;
        font-style: italic;
    }
    .batch-card-filename {
        font-size: 0.68rem;
        color: #2A2200;
        margin-top: 4px;
        font-family: 'EB Garamond', serif;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Summary stat box ── */
    .stat-box {
        background: #0D0D0D;
        border: 1px solid #1A1400;
        border-radius: 2px;
        padding: 20px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .stat-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #C9A84C55, transparent);
    }
    .stat-number {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 300;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .stat-label {
        font-family: 'Cinzel', serif;
        font-size: 0.6rem;
        color: #5C4A1A;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 6px;
    }

    /* ── Upload area ── */
    div[data-testid="stFileUploader"] {
        background: #0D0D0D;
        border: 1px dashed #2A2200;
        border-radius: 4px;
        padding: 10px;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #C9A84C55;
    }

    /* ── Dataset sample card ── */
    .ds-card {
        background: #0D0D0D;
        border: 1px solid #1A1400;
        border-radius: 2px;
        overflow: hidden;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .ds-card:hover { transform: translateY(-3px); border-color: #C9A84C; }
    .ds-card-label {
        padding: 8px 10px;
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .ds-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 0;
        font-size: 0.65rem;
        font-family: 'Cinzel', serif;
        font-weight: 600;
        letter-spacing: 0.08em;
    }

    /* ── Metric card (about page) ── */
    .tech-chip {
        background: #0D0D0D;
        border: 1px solid #2A2200;
        border-radius: 2px;
        padding: 18px;
        text-align: center;
        font-family: 'Cinzel', serif;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        color: #C9A84C;
        text-transform: uppercase;
        transition: background 0.3s, box-shadow 0.3s;
    }
    .tech-chip:hover {
        background: #111100;
        box-shadow: 0 4px 20px rgba(201,168,76,0.08);
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: #080808 !important;
        border-right: 1px solid #1A1400;
    }

    /* ── HR divider ── */
    hr { border-color: #1A1400 !important; }

    /* ── Streamlit widget text ── */
    .stMarkdown p { color: #8A7040; }
    label { color: #7A6030 !important; font-family: 'Cinzel', serif !important; font-size: 0.75rem !important; letter-spacing: 0.1em !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; background: #080808; }
    ::-webkit-scrollbar-thumb { background: #2A2200; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-family:Cinzel,serif;font-size:1.1rem;letter-spacing:0.18em;color:#C9A84C;text-transform:uppercase;margin-bottom:2px;'>Vehicle</p><p style='font-family:Cinzel,serif;font-size:0.65rem;letter-spacing:0.3em;color:#5C4A1A;text-transform:uppercase;margin-top:0;'>Recognition System</p>", unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio(
        "Mode",
        ["🏠 Home", "📷 Image", "🖼️ Batch", "📈 Training Curves", "🗂️ Dataset", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<span style='font-family:Cinzel,serif;font-size:0.65rem;letter-spacing:0.15em;color:#5C4A1A;text-transform:uppercase;'>Model:</span> <span style='color:#C9A84C;font-family:EB Garamond,serif;font-size:0.9rem;'>YOLOv8s-cls</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-family:Cinzel,serif;font-size:0.65rem;letter-spacing:0.15em;color:#5C4A1A;text-transform:uppercase;'>Classes:</span>", unsafe_allow_html=True)
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
        color = CLASS_COLORS.get(cls, '#5C4A1A')
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
st.markdown("<h1 style='font-family:Cinzel,serif;font-size:1.9rem;font-weight:600;color:#C9A84C;letter-spacing:0.12em;margin-bottom:2px;'>Vehicle Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family:Cinzel,serif;color:#5C4A1A;font-size:0.6rem;letter-spacing:0.3em;text-transform:uppercase;'>YOLOv8 &nbsp;·&nbsp; Bus &nbsp;·&nbsp; Car &nbsp;·&nbsp; Motorcycle &nbsp;·&nbsp; Truck</p>", unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🏠 Home":
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(160deg, #0D0D0D 0%, #0A0A0A 60%, #0D0800 100%);
        border: 1px solid #2A2200;
        border-top: 2px solid #C9A84C;
        border-radius: 4px;
        padding: 64px 56px 56px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -60px; left: -60px;
            width: 280px; height: 280px;
            background: radial-gradient(circle, #C9A84C0A 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute; bottom: -60px; right: -60px;
            width: 280px; height: 280px;
            background: radial-gradient(circle, #8B69140A 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute; top: 0; left: 50%; transform: translateX(-50%);
            width: 120px; height: 1px;
            background: linear-gradient(90deg, transparent, #C9A84C88, transparent);
        "></div>
        <p style="font-size:3.2rem; margin:0 0 20px 0; line-height:1; opacity:0.7;">🚗</p>
        <h1 style="
            font-family:'Cinzel',serif;
            font-size:2.4rem;
            font-weight:700;
            color:#C9A84C;
            margin:0 0 12px 0;
            letter-spacing:0.14em;
            text-transform:uppercase;
        ">Vehicle Recognition System</h1>
        <p style="
            color:#5C4A1A;
            font-family:'Cinzel',serif;
            font-size:0.62rem;
            letter-spacing:0.3em;
            text-transform:uppercase;
            margin:0 0 28px 0;
        ">YOLOv8S-CLS &nbsp;&middot;&nbsp; BUS &nbsp;&middot;&nbsp; CAR &nbsp;&middot;&nbsp; MOTORCYCLE &nbsp;&middot;&nbsp; TRUCK</p>
        <p style="color:#7A6030; font-family:'EB Garamond',serif; font-size:1.1rem; max-width:520px; margin:0 auto; line-height:1.8; font-style:italic;">
            A deep learning system that classifies vehicle images into four categories
            in real time using a fine-tuned YOLOv8s classification model.
        </p>
        <div style="
            position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
            width: 80px; height: 1px;
            background: linear-gradient(90deg, transparent, #C9A84C44, transparent);
        "></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-stats row ───────────────────────────────────────────────────────
    qs = st.columns(4)
    quick_stats = [
        ("8,863",  "Total Images",    "#C9A84C"),
        ("4",      "Classes",         "#E2C07A"),
        ("224px",  "Input Size",      "#D4AF37"),
        ("70/15/15", "Train/Val/Test", "#8B6914"),
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
        ("📷", "Image",          "Upload a single image and get an instant classification with confidence scores and probability bars.", "#C9A84C"),
        ("🖼️", "Batch",          "Predict on multiple images at once. Export results to CSV and see class distribution at a glance.",   "#E2C07A"),
        ("📈", "Training Curves","Upload your YOLOv8 results.csv to visualise loss, accuracy, and learning-rate curves over epochs.",    "#D4AF37"),
        ("🗂️", "Dataset",        "Explore the dataset composition — class counts, train/val/test splits, and proportion charts.",        "#8B6914"),
    ]
    for col, (icon, title, desc, color) in zip(fc, features):
        col.markdown(f"""
        <div style="
            background: #0D0D0D;
            border: 1px solid #1A1400;
            border-top: 2px solid {color};
            border-radius: 2px;
            padding: 24px 18px;
            height: 100%;
            transition: transform 0.3s, box-shadow 0.3s;
        ">
            <p style="font-size:1.6rem; margin:0 0 12px 0; opacity:0.8;">{icon}</p>
            <p style="
                font-family:'Cinzel',serif;
                font-size:0.75rem;
                font-weight:600;
                color:{color};
                margin:0 0 10px 0;
                letter-spacing:0.12em;
                text-transform:uppercase;
            ">{title}</p>
            <p style="color:#5C4A1A; font-family:'EB Garamond',serif; font-size:0.95rem; line-height:1.7; margin:0; font-style:italic;">{desc}</p>
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
            background:#0D0D0D;
            border:1px solid #1A1400;
            border-top:2px solid {color};
            border-radius:2px;
            padding:24px 16px;
            text-align:center;
        ">
            <p style="font-size:2.2rem; margin:0 0 10px 0; opacity:0.8;">{icon}</p>
            <p style="
                font-family:'Cinzel',serif;
                font-size:0.75rem;
                font-weight:600;
                color:{color};
                margin:0 0 10px 0;
                letter-spacing:0.14em;
                text-transform:uppercase;
            ">{cls}</p>
            <p style="color:#5C4A1A; font-family:'EB Garamond',serif; font-size:0.9rem; line-height:1.65; margin:0; font-style:italic;">{class_descs[cls]}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">How it works</p>', unsafe_allow_html=True)
    steps = [
        ("I",  "Upload",     "Drop an image or batch of images into the app.",              "#C9A84C"),
        ("II", "Preprocess", "Frames are converted to grayscale and resized to 224×224.",   "#D4AF37"),
        ("III","Infer",      "YOLOv8s-cls runs a forward pass and outputs class logits.",   "#E2C07A"),
        ("IV", "Result",     "Softmax probabilities are ranked and displayed with labels.", "#8B6914"),
    ]
    pipe_cols = st.columns(len(steps))
    for col, (num, title, desc, color) in zip(pipe_cols, steps):
        col.markdown(f"""
        <div style="
            background:#0D0D0D;
            border:1px solid #1A1400;
            border-radius:2px;
            padding:20px 14px;
            text-align:center;
            position:relative;
        ">
            <p style="
                font-family:'Cormorant Garamond',serif;
                font-size:2.8rem;
                font-weight:300;
                color:{color}22;
                margin:0 0 6px 0;
                line-height:1;
                font-style:italic;
            ">{num}</p>
            <p style="
                font-family:'Cinzel',serif;
                font-size:0.72rem;
                font-weight:600;
                color:{color};
                margin:0 0 8px 0;
                letter-spacing:0.12em;
                text-transform:uppercase;
            ">{title}</p>
            <p style="color:#5C4A1A; font-family:'EB Garamond',serif; font-size:0.9rem; line-height:1.6; margin:0; font-style:italic;">{desc}</p>
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
                color = CLASS_COLORS.get(cls_name, '#5C4A1A')
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
        <div style="text-align:center; padding: 60px 20px; color:#2A2200;">
            <p style="font-size:3rem">📷</p>
            <p style="font-family:'Cinzel',serif; font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase;">UPLOAD AN IMAGE TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🖼️ Batch":
    st.markdown("### Batch Prediction")
    st.markdown("<p style='color:#5C4A1A;'>Upload multiple vehicle images and get predictions for all at once.</p>", unsafe_allow_html=True)

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
                <div class="stat-number" style="color:#C9A84C">{val}</div>
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
                color = CLASS_COLORS.get(cls, '#5C4A1A')
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
        <div style="text-align:center; padding: 60px 20px; color:#2A2200;">
            <p style="font-size:3rem">🖼️</p>
            <p style="font-family:'Cinzel',serif; font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase;">UPLOAD MULTIPLE IMAGES TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# MODE: TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Training Curves":
    st.markdown("### Training Curves")
    st.markdown(
        "<p style='color:#5C4A1A;'>Upload your <code>results.csv</code> exported by YOLOv8 training to visualise training dynamics.</p>",
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
                    ax.plot(epochs, detected[lr_keys[0]], linewidth=1.6, color="#C9A84C")
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
                        <div class="stat-number" style="color:#C9A84C;font-size:1.1rem;">{v}</div>
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
        <div style="background:#0D0D0D;border:1px solid #1A1400;border-radius:2px;padding:24px;margin-top:16px;">
            <p style="font-family:'Cinzel',serif;color:#5C4A1A;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">Expected Columns</p>
            <ul style="color:#C9A84C;line-height:2;">
                <li><code>epoch</code></li>
                <li><code>train/box_loss</code>, <code>train/cls_loss</code></li>
                <li><code>val/box_loss</code>, <code>val/cls_loss</code></li>
                <li><code>metrics/accuracy_top1</code>, <code>metrics/accuracy_top5</code></li>
                <li><code>lr/pg0</code></li>
            </ul>
            <p style="color:#5C4A1A;font-size:0.85rem;">
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
    st.markdown("<p style='color:#5C4A1A;'>Class distribution across the training split (~4,081 images).</p>", unsafe_allow_html=True)
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
    stat_colors = ["#E2C07A", CLASS_COLORS['Bus'], CLASS_COLORS['Car'], CLASS_COLORS['Motorcycle'], CLASS_COLORS['Truck']]
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
        "Train": (0.70, "#C9A84C"),
        "Val":   (0.15, "#E2C07A"),
        "Test":  (0.15, "#8B6914"),
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
        legend_html += f"<span style='font-size:0.75rem;color:{color};font-family:'Cinzel',serif;font-size:0.65rem;letter-spacing:0.15em;'>■ {split_name} {int(ratio*100)}%</span>"
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
