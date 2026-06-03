# 🚗 Vehicle Recognition System — Streamlit App

AI-powered vehicle classifier (Bus / Car / Motorcycle / Truck) using YOLOv8.

## 📁 File Structure

```
your-repo/
├── app.py                  ← Main Streamlit app
├── best.pt                 ← Your trained YOLOv8 model weights  ⚠️ ADD THIS
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System dependencies (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml         ← Theme & server settings
└── README.md
```

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
1. Create a new GitHub repository (public or private).
2. Add all files from this folder to the repo.
3. **Copy your trained `best.pt`** file into the repo root.
   - If the file is >100 MB, use [Git LFS](https://git-lfs.github.com/):
     ```bash
     git lfs install
     git lfs track "*.pt"
     git add .gitattributes
     git add best.pt
     git commit -m "Add model weights"
     git push
     ```

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set the **Main file path** to `app.py`.
4. Click **Deploy**.

That's it — Streamlit Cloud installs `requirements.txt` and `packages.txt` automatically.

---

## 💻 Run Locally

```bash
# 1. Clone your repo or cd into the project folder
cd vehicle-recognition-app

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure best.pt is in the same folder

# 5. Launch
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## ⚠️ Important Notes

| Note | Detail |
|------|--------|
| Model file | `best.pt` must be in the **repo root** alongside `app.py` |
| File size limit | Streamlit Cloud has a 1 GB memory limit; YOLOv8s is ~22 MB so it's fine |
| Upload limit | Set to 200 MB in `config.toml`; adjust if needed |
| OpenCV | `opencv-python-headless` is used (no GUI dependency — required for cloud) |
| Webcam | Live webcam mode from the notebook is **not supported** on hosted Streamlit; use the Image or Video modes instead |

---

## 🎛️ App Features

- **Image mode** — upload any JPG/PNG and get instant class prediction with confidence bars
- **Video mode** — upload an MP4/AVI; the app processes every frame and gives you a labelled video to download
- **Confidence threshold** — sidebar slider to filter low-confidence predictions
- **About** — model details and tech stack

---

## 🛠️ Customising

- **Add more classes**: update `CLASSES`, `CLASS_COLORS`, and `CLASS_ICONS` at the top of `app.py`.
- **Change model**: replace `best.pt` with any YOLOv8 classification `.pt` file.
- **Theme**: edit `.streamlit/config.toml`.
