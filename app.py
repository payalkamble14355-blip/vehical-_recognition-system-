import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
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
    'Bus':        '#FF8C00',
    'Car':        '#2ECC71',
    'Motorcycle': '#3498DB',
    'Truck':      '#E74C3C',
}
CLASS_ICONS = {
    'Bus': '🚌',
    'Car': '🚗',
    'Motorcycle': '🏍️',
    'Truck': '🚛',
}

# ── Plot theme helper ──────────────────────────────────────────────────────────
PLT_BG       = "#0f1117"
PLT_SURFACE  = "#1a1d2e"
PLT_BORDER   = "#2d3561"
PLT_TEXT     = "#c8cfe8"
PLT_SUBTEXT  = "#5a6080"

CURVE_COLORS = {
    "train": "#3498DB",
    "val":   "#E74C3C",
    "top1":  "#2ECC71",
    "top3":  "#FF8C00",
    "bus":   "#FF8C00",
    "car":   "#2ECC71",
    "motorcycle": "#3498DB",
    "truck": "#E74C3C",
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
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    /* ── Prediction card ── */
    .pred-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        text-align: center;
        animation: fadeSlideIn 0.4s ease;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .pred-label {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .pred-conf {
        font-size: 1.1rem;
        color: #a0a8c0;
        margin-top: 4px;
    }

    /* ── Probability bars ── */
    .prob-bar-container {
        background: #1a1d2e;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2d3561;
        margin: 8px 0;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 8px 0;
        gap: 10px;
    }
    .prob-label {
        width: 110px;
        font-size: 0.9rem;
        color: #c8cfe8;
        font-weight: 500;
    }
    .prob-track {
        flex: 1;
        height: 10px;
        background: #2d3561;
        border-radius: 5px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.6s ease;
    }
    .prob-val {
        width: 48px;
        text-align: right;
        font-size: 0.85rem;
        color: #7b84a8;
        font-family: 'Space Mono', monospace;
    }

    /* ── Section header ── */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #5a6080;
        margin-bottom: 12px;
    }

    /* ── Batch result card ── */
    .batch-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .batch-card:hover {
        transform: translateY(-3px);
        border-color: #4a5890;
    }
    .batch-card-icon { font-size: 2rem; margin-bottom: 6px; }
    .batch-card-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .batch-card-conf {
        font-size: 0.78rem;
        color: #7b84a8;
        margin-top: 2px;
    }
    .batch-card-filename {
        font-size: 0.7rem;
        color: #3a4060;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Summary stat box ── */
    .stat-box {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #5a6080;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ── Upload area ── */
    div[data-testid="stFileUploader"] {
        background: #1a1d2e;
        border: 2px dashed #2d3561;
        border-radius: 12px;
        padding: 10px;
    }

    /* ── Metric card (about page) ── */
    .tech-chip {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #c8cfe8;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Vehicle AI")
    st.markdown("---")
    mode = st.radio(
        "Mode",
        ["📷 Image", "🖼️ Batch", "🎬 Video", "📈 Training Curves", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** YOLOv8s-cls")
    st.markdown("**Classes:**")
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
st.markdown("# 🚗 Vehicle Recognition System")
st.markdown("<p style='color:#5a6080;font-family:Space Mono,monospace;font-size:0.8rem;letter-spacing:0.1em;'>YOLOV8 · BUS · CAR · MOTORCYCLE · TRUCK</p>", unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📷 Image":
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
            <p style="font-family:'Space Mono',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD AN IMAGE TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🖼️ Batch":
    st.markdown("### Batch Prediction")
    st.markdown("<p style='color:#5a6080;'>Upload multiple vehicle images and get predictions for all at once.</p>", unsafe_allow_html=True)

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
                <div class="stat-number" style="color:#c8cfe8">{val}</div>
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
            <p style="font-family:'Space Mono',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD MULTIPLE IMAGES TO BEGIN</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: VIDEO
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🎬 Video":
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if uploaded_video:
        model = load_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

        output_path = tmp_in_path.replace(".mp4", "_output.mp4")

        cap = cv2.VideoCapture(tmp_in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        FONT = cv2.FONT_HERSHEY_DUPLEX
        CLASS_COLORS_BGR = {
            'Bus':        (50, 140, 255),
            'Car':        (50, 200, 50),
            'Motorcycle': (219, 152, 52),
            'Truck':      (52, 94, 235),
        }

        st.markdown("**Processing video...**")
        progress = st.progress(0)
        status = st.empty()
        count = 0
        frame_log = []  # track per-frame predictions for a timeline

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            gray3 = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            result = model.predict(gray3, imgsz=224, verbose=False)[0]
            probs_arr = result.probs.data.cpu().numpy()
            pred_idx = int(probs_arr.argmax())
            pred_conf = float(probs_arr.max())
            pred_name = model.names[pred_idx]
            label = f"{pred_name}  {pred_conf*100:.1f}%"
            color = CLASS_COLORS_BGR.get(pred_name, (200, 200, 200))

            cv2.putText(frame, label, (20, 50), FONT, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (20, 50), FONT, 1.2, color, 2, cv2.LINE_AA)
            out.write(frame)
            frame_log.append((count / fps, pred_name, pred_conf))

            if count % 10 == 0:
                progress.progress(min(count / max(total, 1), 1.0))
                status.text(f"Frame {count}/{total}")

        cap.release()
        out.release()
        os.unlink(tmp_in_path)
        progress.progress(1.0)
        status.text(f"✅ Done! Processed {count} frames.")

        # ── Post-video stats ──────────────────────────────────────────────────
        if frame_log:
            st.markdown("---")
            st.markdown('<p class="section-header">Video Analysis</p>', unsafe_allow_html=True)

            times   = [x[0] for x in frame_log]
            preds   = [x[1] for x in frame_log]
            confs   = [x[2] for x in frame_log]

            # Confidence timeline
            fig, ax = plt.subplots(figsize=(8, 2.5))
            for cls in CLASSES:
                mask = [i for i, p in enumerate(preds) if p == cls]
                if mask:
                    ax.scatter([times[i] for i in mask], [confs[i] for i in mask],
                               s=8, label=cls, color=CLASS_COLORS[cls], alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Confidence")
            ax.set_title("Per-Frame Confidence Timeline")
            ax.legend(fontsize=8, labelcolor=PLT_TEXT, facecolor=PLT_SURFACE, edgecolor=PLT_BORDER)
            apply_dark_theme(fig, [ax])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed Video",
                f,
                file_name="vehicle_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        os.unlink(output_path)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color:#3a4060;">
            <p style="font-size:3rem">🎬</p>
            <p style="font-family:'Space Mono',monospace; font-size:0.9rem; letter-spacing:0.1em;">UPLOAD A VIDEO TO PROCESS</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📈 Training Curves":
    st.markdown("### Training Curves")
    st.markdown(
        "<p style='color:#5a6080;'>Upload your <code>results.csv</code> exported by YOLOv8 training to visualise training dynamics.</p>",
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
                    ax.plot(epochs, detected[lr_keys[0]], linewidth=1.6, color="#9B59B6")
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
                        <div class="stat-number" style="color:#c8cfe8;font-size:1.1rem;">{v}</div>
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
            <p style="font-family:'Space Mono',monospace;color:#5a6080;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">Expected Columns</p>
            <ul style="color:#c8cfe8;line-height:2;">
                <li><code>epoch</code></li>
                <li><code>train/box_loss</code>, <code>train/cls_loss</code></li>
                <li><code>val/box_loss</code>, <code>val/cls_loss</code></li>
                <li><code>metrics/accuracy_top1</code>, <code>metrics/accuracy_top5</code></li>
                <li><code>lr/pg0</code></li>
            </ul>
            <p style="color:#5a6080;font-size:0.85rem;">
                YOLOv8 saves this file automatically to <code>runs/classify/train/results.csv</code>
                after training completes.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "ℹ️ About":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Project Summary")
        st.markdown("""
        This system uses **YOLOv8s-cls** fine-tuned on a vehicle dataset to classify
        images and video frames into four categories in real time.

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
