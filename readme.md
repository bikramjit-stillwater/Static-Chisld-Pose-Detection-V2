# Tadasana Pose Detection and Feedback

A web application that analyzes Tadasana (Mountain Pose) videos using MediaPipe for
pose detection and Google Gemini for natural-language feedback. Each video is scored
across 6 ground-truth alignment principles, with cropped per-step images and a final
recommendation.

## Features

- 6-step rule-based scoring (Stance, Body Balance, Legs & Knees, Spine,
  Shoulders & Arms, Head & Neck)
- Per-step zoomed images with green/red skeleton overlay
- LLM-generated coaching feedback (Gemini 2.5 Flash)
- Streamlit web UI

## Project Structure

```
tadasana-app/
├── app.py                      # Streamlit UI entry point
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── .python-version             # Python version pin
├── .gitignore                  # Files excluded from git
├── .env                        # Local secrets (NOT committed)
└── src/
    ├── __init__.py
    ├── pose_detector.py        # MediaPipe wrapper
    ├── pose_analyzer.py        # Frame loop, features, image generation
    ├── scorer.py               # 6-step rule engine
    ├── feedback.py             # Gemini feedback + rule-based fallback
    └── recorder.py             # OpenCV recording helper
```

## Local Development

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd tadasana-app
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
# or
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a local .env file

```
GEMINI_API_KEY=your_actual_key_here
```

This file is gitignored and stays on your machine.

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deployment to Render

### Step 1 - Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### Step 2 - Create the Render service

1. Go to https://dashboard.render.com
2. Click **New +** -> **Web Service**
3. Connect your GitHub repo
4. Render will automatically detect `render.yaml` and pre-fill the settings
5. Click **Create Web Service**

### Step 3 - Add the Gemini API key

1. Inside the new service, open the **Environment** tab
2. Add a variable:
   - Key: `GEMINI_API_KEY`
   - Value: your actual Gemini API key
3. Click **Save Changes**

The service will redeploy automatically with the key available.

### Step 4 - Visit your live URL

Render gives you a URL like `https://tadasana-pose-app.onrender.com`. Open it and
test by uploading a video.

## Notes on the Free Tier

- Free Render services sleep after 15 minutes of inactivity. The first request
  after sleeping takes 30-60 seconds to wake the container.
- Free tier has 512MB RAM. MediaPipe + OpenCV will work but may slow down on
  large videos. Consider the **starter** plan ($7/month) for production use.
- Render's filesystem is ephemeral - any files written at runtime
  (output/recordings/, output/extracted_frames/) will be wiped between deploys.
  This is fine for a stateless analysis app.

## Environment Variables Reference

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | No | Enables LLM feedback. If unset, app uses rule-based fallback. |
| `PYTHON_VERSION` | No | Set automatically via render.yaml. |
| `PORT` | Auto | Set by Render at runtime. |

## Tech Stack

- **MediaPipe BlazePose** - body landmark detection (local CNN)
- **OpenCV** - frame iteration, drawing, image cropping
- **Streamlit** - web UI
- **Google Gemini 2.5 Flash** - feedback generation (cloud LLM)
- **Render** - hosting platform