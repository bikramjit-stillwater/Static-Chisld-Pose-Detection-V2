import os
import tempfile
from datetime import datetime
import streamlit as st

from src.pose_analyzer import analyze_video, analyze_image
from src.feedback import get_gemini_feedback


st.set_page_config(
    page_title="Extended Child's Pose Analysis",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
section.main > div.block-container {
    padding-top: 2rem; padding-bottom: 3rem;
    padding-left: 2rem; padding-right: 2rem;
    max-width: 1400px;
}
* { box-sizing: border-box; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {height: 0; visibility: hidden;}
.stDeployButton {display: none;}

/* =====================================================================
   PAGE-LEVEL TEXT - all white/light because page background is dark
   ===================================================================== */
.page-header {
    margin-bottom: 1.5rem; padding-bottom: 1rem;
    border-bottom: 2px solid #334155;
}
.page-header h1 {
    font-size: 1.75rem !important; font-weight: 700 !important;
    color: #FFFFFF !important; margin: 0 !important;
    line-height: 1.2; letter-spacing: -0.02em;
}
.page-header .subtitle {
    font-size: 0.875rem;
    color: #cbd5e1;
    margin-top: 0.35rem;
}

/* Section headings - white on dark page */
.stMarkdown h2, h2 {
    font-size: 1.15rem !important; font-weight: 600 !important;
    color: #FFFFFF !important;
    margin-top: 1.5rem !important; margin-bottom: 0.75rem !important;
}
.stMarkdown h3, h3 {
    font-size: 0.95rem !important; font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-top: 0.5rem !important; margin-bottom: 0.4rem !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 0.875rem !important; line-height: 1.5 !important;
    color: #e2e8f0 !important;
}
.stMarkdown strong { color: #FFFFFF !important; }

/* Captions - light on dark */
[data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.4rem;
}

/* =====================================================================
   SIDE-VIEW INSTRUCTION BANNER (NEW - prominent at top)
   ===================================================================== */
.side-view-banner {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    color: white;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25);
    border: 1px solid #3b82f6;
}
.side-view-banner .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #bfdbfe;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.side-view-banner .title {
    font-size: 1.05rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.3rem;
}
.side-view-banner .body-text {
    font-size: 0.85rem;
    color: #dbeafe;
    line-height: 1.5;
}
.side-view-banner ul { margin: 0.4rem 0 0 1.2rem; padding: 0; }
.side-view-banner li { margin: 0.15rem 0; color: #dbeafe; }

/* =====================================================================
   SCORE HERO - already light text on dark gradient, fine
   ===================================================================== */
.score-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 14px; padding: 1.5rem;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    height: 100%; display: flex; flex-direction: column;
    justify-content: space-between; min-height: 380px;
    border: 1px solid #1e293b;
}
.score-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem;
}
.score-value {
    font-size: 4rem; font-weight: 800; line-height: 1;
    color: white; letter-spacing: -0.04em;
}
.score-suffix { font-size: 1.5rem; color: #94a3b8; font-weight: 400; }
.score-band {
    display: inline-block; padding: 0.25rem 0.7rem; border-radius: 100px;
    font-size: 0.7rem; font-weight: 600; margin-top: 0.6rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.band-excellent { background: #10b981; color: white; }
.band-good      { background: #3b82f6; color: white; }
.band-mixed     { background: #f59e0b; color: white; }
.band-poor      { background: #ef4444; color: white; }

.input-mode-tag {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 100px;
    font-size: 0.65rem;
    font-weight: 600;
    background: rgba(255,255,255,0.15);
    color: #e2e8f0;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.issues-block { margin-top: 1.5rem; }
.issues-block h4 {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; font-weight: 600; margin: 0 0 0.6rem 0;
}
.issue-item {
    background: rgba(255,255,255,0.08); border-left: 3px solid #fbbf24;
    padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 4px;
    font-size: 0.82rem; color: #e2e8f0; line-height: 1.4;
}
.no-issues {
    background: rgba(16,185,129,0.15); border-left: 3px solid #10b981;
    padding: 0.5rem 0.75rem; border-radius: 4px;
    font-size: 0.82rem; color: #d1fae5;
}

/* =====================================================================
   IMAGE CARDS - dark theme version (was white before)
   ===================================================================== */
.image-card {
    background: #1e293b;
    border-radius: 14px; padding: 1rem;
    border: 1px solid #334155;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    height: 100%; min-height: 380px;
    display: flex; flex-direction: column;
}
.image-card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.6rem;
}
.image-card-title {
    font-size: 0.95rem; font-weight: 600;
    color: #FFFFFF;
}
.image-card-legend {
    font-size: 0.7rem;
    color: #94a3b8;
}
.legend-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 0.3rem; vertical-align: middle;
}

.annotated-wrap {
    flex: 1; display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}
.annotated-wrap [data-testid="stImage"] { width: auto !important; max-width: 100% !important; }
.annotated-wrap [data-testid="stImage"] img {
    max-height: 340px !important; width: auto !important;
    object-fit: contain !important; border-radius: 8px;
    margin: 0 auto; display: block;
}

/* =====================================================================
   STEP CARDS - dark theme (white text on dark card background)
   ===================================================================== */
.step-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px; padding: 0.85rem; margin-bottom: 0.75rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    transition: all 0.15s ease;
    height: 100%; display: flex; flex-direction: column;
}
.step-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    transform: translateY(-1px);
}
.step-card.passed { border-left: 3px solid #10b981; }
.step-card.failed { border-left: 3px solid #ef4444; }
.step-card.notvisible {
    border-left: 3px solid #9ca3af;
    background: #0f172a;
}

.step-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.5rem;
}
.step-name {
    font-size: 0.9rem; font-weight: 600;
    color: #FFFFFF;
    line-height: 1.2;
}
.step-card.notvisible .step-name { color: #94a3b8; }
.step-status-icon { font-size: 1.1rem; }

.step-score-row {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.6rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}
.step-score-big { font-size: 1.4rem; font-weight: 700; line-height: 1; }
.step-score-big.pass { color: #10b981; }
.step-score-big.fail { color: #ef4444; }
.step-score-big.notvisible { color: #9ca3af; }
.step-score-suffix { font-size: 0.75rem; color: #94a3b8; }
.step-fail-rate { font-size: 0.7rem; color: #94a3b8; }

.step-image-wrap {
    width: 100%; height: 180px;
    display: flex; align-items: center; justify-content: center;
    background: #0f172a;
    border-radius: 6px; overflow: hidden;
    margin-bottom: 0.5rem;
}
.step-image-wrap [data-testid="stImage"] {
    width: 100% !important; height: 100% !important;
    display: flex !important; align-items: center !important;
    justify-content: center !important;
}
.step-image-wrap [data-testid="stImage"] img {
    max-height: 180px !important; max-width: 100% !important;
    width: auto !important; object-fit: contain !important;
}

.step-issue {
    background: rgba(239,68,68,0.15);
    border-left: 2px solid #ef4444;
    padding: 0.45rem 0.6rem; border-radius: 4px;
    font-size: 0.75rem; color: #fecaca; line-height: 1.4;
    margin-bottom: 0.35rem;
}
.step-issue strong { color: #FFFFFF; }
.step-passed-msg {
    background: rgba(16,185,129,0.15);
    border-left: 2px solid #10b981;
    padding: 0.45rem 0.6rem; border-radius: 4px;
    font-size: 0.75rem; color: #d1fae5; margin-bottom: 0.35rem;
}
.step-passed-msg strong { color: #FFFFFF; }
.step-notvisible-msg {
    background: rgba(156,163,175,0.15);
    border-left: 2px solid #9ca3af;
    padding: 0.45rem 0.6rem; border-radius: 4px;
    font-size: 0.75rem; color: #d1d5db; line-height: 1.4;
    margin-bottom: 0.35rem;
}
.step-notvisible-msg strong { color: #FFFFFF; }
.step-cue {
    background: rgba(59,130,246,0.12);
    border-left: 2px solid #3b82f6;
    padding: 0.45rem 0.6rem; border-radius: 4px;
    font-size: 0.72rem; color: #bfdbfe; line-height: 1.4;
    font-style: italic;
}
.step-cue strong { font-style: normal; color: #FFFFFF; }

/* =====================================================================
   UPLOAD CONTROLS
   ===================================================================== */
[data-testid="stFileUploader"] section {
    padding: 1rem !important;
    border: 1.5px dashed #475569; border-radius: 8px;
    background-color: #1e293b; transition: border-color 0.15s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #3b82f6; background-color: #334155;
}

.stButton > button {
    background-color: #2563eb; color: white; font-weight: 600;
    padding: 0.55rem 1.5rem; border-radius: 8px; border: none;
    font-size: 0.875rem; transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.stButton > button:hover {
    background-color: #1d4ed8; color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.35);
}
.stButton > button[kind="primary"] { background-color: #2563eb; }
.stButton > button[kind="primary"]:hover { background-color: #1d4ed8; }

[data-testid="stVideo"] {
    border-radius: 8px; overflow: hidden;
    max-width: 480px; margin-bottom: 0.5rem;
}
[data-testid="stVideo"] video { max-height: 360px; border-radius: 8px; }

.preview-image [data-testid="stImage"] img {
    max-height: 360px !important; max-width: 100% !important;
    width: auto !important; object-fit: contain !important;
    border-radius: 8px;
}

[data-testid="stAlert"] {
    padding: 0.55rem 0.85rem !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.4 !important;
}

.stTextArea textarea {
    font-size: 0.85rem !important; line-height: 1.6 !important;
    background-color: #1e293b !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    padding: 0.85rem !important;
    color: #e2e8f0 !important;
}

hr {
    margin: 1.25rem 0 !important; border: none !important;
    border-top: 1px solid #334155 !important;
}

@media (max-width: 768px) {
    .page-header h1 { font-size: 1.4rem !important; }
    .score-value { font-size: 3rem; }
    .score-hero, .image-card { min-height: auto; }
    .step-image-wrap { height: 150px; }
    .annotated-wrap [data-testid="stImage"] img { max-height: 280px !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_output_dir(subdir):
    base = os.environ.get("OUTPUT_DIR")
    if base:
        path = os.path.join(base, subdir)
    else:
        local = os.path.join("output", subdir)
        try:
            os.makedirs(local, exist_ok=True)
            test = os.path.join(local, ".write_test")
            with open(test, "w") as f: f.write("ok")
            os.remove(test)
            path = local
        except OSError:
            path = os.path.join(tempfile.gettempdir(), "childpose", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def score_band(score):
    if score is None: return ("Not scored", "band-poor")
    if score >= 85: return ("Excellent", "band-excellent")
    if score >= 70: return ("Good", "band-good")
    if score >= 50: return ("Mixed", "band-mixed")
    return ("Needs work", "band-poor")


def is_video(path):
    return path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def is_image(path):
    return path.lower().endswith((".jpg", ".jpeg", ".png"))


# Header
st.markdown("""
<div class='page-header'>
  <h1>🧘 Extended Child's Pose Analysis</h1>
  <div class='subtitle'>Utthita Balasana - Upload a video OR take a photo. Get a step-by-step alignment report.</div>
</div>
""", unsafe_allow_html=True)

# SIDE-VIEW BANNER - prominent at top, always visible
st.markdown("""
<div class='side-view-banner'>
  <div class='label'>📐 Important - record from the SIDE</div>
  <div class='title'>This pose is on the floor - capture from the side of your body</div>
  <div class='body-text'>
    For the analysis to work, your full body must be visible from a side angle:
    <ul>
      <li>Place your camera at floor / mat level, beside your body</li>
      <li>The whole body should fit in the frame - knees, hips, torso, arms, head</li>
      <li>Do NOT shoot from above (top-down) or from the front</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)

recordings_dir = get_output_dir("recordings")
frames_dir = get_output_dir("extracted_frames")

for key, default in [
    ("video_path", None),
    ("input_mode", None),
    ("analysis_result", None),
    ("gemini_feedback", None),
    ("capture_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Upload section
if not st.session_state.analysis_result:
    st.markdown("## Upload Your Pose")

    in1, in2 = st.columns(2)
    with in1:
        st.markdown("**Option 1 — Upload video**")
        uploaded_video = st.file_uploader(
            "video", type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed",
        )
        st.caption("Side view of your body, knees on the mat, arms reaching forward.")
    with in2:
        st.markdown("**Option 2 — Take a photo**")
        camera_file = st.camera_input("camera", label_visibility="collapsed")
        st.caption("Position camera at floor level, beside your body.")
else:
    uploaded_video = None
    camera_file = None
    if st.button("← Analyze a different one"):
        for k in ["video_path", "input_mode", "analysis_result",
                  "gemini_feedback", "capture_done"]:
            st.session_state[k] = None
        st.rerun()

if uploaded_video is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(recordings_dir, f"uploaded_{timestamp}.mp4")
    with open(save_path, "wb") as f:
        f.write(uploaded_video.read())
    st.session_state.video_path = save_path
    st.session_state.input_mode = "video"
    st.session_state.capture_done = True
    st.success("✓ Video uploaded successfully.")

if camera_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(frames_dir, f"captured_pose_{timestamp}.jpg")
    with open(image_path, "wb") as f:
        f.write(camera_file.read())
    st.session_state.video_path = image_path
    st.session_state.input_mode = "photo"
    st.session_state.capture_done = True
    st.success("✓ Photo captured successfully.")

# Preview + Analyze button
if st.session_state.video_path and not st.session_state.analysis_result:
    st.markdown("### Preview")
    pv1, _ = st.columns([2, 1])
    with pv1:
        if is_video(st.session_state.video_path):
            st.video(st.session_state.video_path)
        else:
            st.markdown("<div class='preview-image'>", unsafe_allow_html=True)
            st.image(st.session_state.video_path)
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Analyze Pose", type="primary"):
        with st.spinner("Analyzing pose and generating feedback..."):
            try:
                if is_image(st.session_state.video_path):
                    result = analyze_image(st.session_state.video_path, frames_dir)
                else:
                    result = analyze_video(st.session_state.video_path, frames_dir)

                feedback_text = get_gemini_feedback(
                    result["final_score"], result["issues"],
                    steps=result.get("steps"),
                )
                st.session_state.analysis_result = result
                st.session_state.gemini_feedback = feedback_text
                st.rerun()
            except Exception as e:
                st.session_state.analysis_result = {
                    "final_score": 0, "issues": [f"Analysis failed: {str(e)}"],
                    "steps": [], "best_frame_path": None,
                    "annotated_path": None, "step_image_paths": {},
                    "low_quality_warning": True,
                    "low_quality_message": f"Analysis failed: {str(e)}",
                }
                st.session_state.gemini_feedback = (
                    f"Could not generate full pose feedback.\n\nReason: {str(e)}"
                )
                st.rerun()

# Results
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    score = result["final_score"]
    band_label, band_class = score_band(score)
    mode = st.session_state.input_mode or "video"

    # Show "not Child's Pose" prominently if pose was rejected
    if result.get("pose_invalid"):
        st.error(
            "❌ **This does not look like Extended Child's Pose.** "
            "You appear to be standing or in a different pose. "
            "Please get into the pose (kneeling, forehead toward the mat, "
            "arms extended forward) and re-record from the SIDE of your body."
        )
        st.markdown("## Coach's Feedback")
        st.text_area(
            "Feedback",
            value=st.session_state.gemini_feedback or "No feedback generated.",
            height=200,
            label_visibility="collapsed",
        )
    else:
        if result.get("low_quality_warning"):
            st.warning(result.get("low_quality_message")
                       or "Low confidence in this analysis - please re-record.")

        # Hero row
        hero_left, hero_right = st.columns([1, 1.5])

        with hero_left:
            issues_html = ""
            if result["issues"]:
                for issue in result["issues"][:3]:
                    issues_html += f"<div class='issue-item'>{issue}</div>"
            else:
                issues_html = "<div class='no-issues'>✓ No major issues detected</div>"

            mode_tag = "📹 Video" if mode == "video" else "📸 Photo"

            st.markdown(f"""
            <div class='score-hero'>
              <div>
                <div class='score-label'>Final Score</div>
                <div>
                  <span class='score-value'>{score}</span>
                  <span class='score-suffix'>/100</span>
                </div>
                <span class='score-band {band_class}'>{band_label}</span>
                <div><span class='input-mode-tag'>{mode_tag}</span></div>
              </div>
              <div class='issues-block'>
                <h4>Top Issues</h4>
                {issues_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with hero_right:
            annotated_path = result.get("annotated_path")
            title = "Annotated Photo" if mode == "photo" else "Annotated Best Frame"
            st.markdown(f"""
            <div class='image-card'>
              <div class='image-card-header'>
                <div class='image-card-title'>{title}</div>
                <div class='image-card-legend'>
                  <span class='legend-dot' style='background:#22c55e'></span>passed
                  &nbsp;
                  <span class='legend-dot' style='background:#ef4444'></span>needs work
                  &nbsp;
                  <span class='legend-dot' style='background:#9ca3af'></span>not visible
                </div>
              </div>
              <div class='annotated-wrap'>
            """, unsafe_allow_html=True)

            if annotated_path and os.path.exists(annotated_path):
                st.image(annotated_path)
            elif result.get("best_frame_path") and os.path.exists(result["best_frame_path"]):
                st.image(result["best_frame_path"])

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Step grid
        steps = result.get("steps") or []
        step_imgs = result.get("step_image_paths") or {}

        if steps:
            st.markdown("## Step-by-Step Breakdown")
            st.caption("Each card zooms into the body part being checked. "
                       "Gray cards mean the body part wasn't visible.")

            for i in range(0, len(steps), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j >= len(steps):
                        continue
                    s = steps[i + j]
                    step_num = s["step"]
                    not_visible = s.get("not_visible", False)
                    with col:
                        if not_visible:
                            st.markdown(f"""
                            <div class='step-card notvisible'>
                              <div class='step-header'>
                                <div class='step-name'>Step {step_num}: {s['name']}</div>
                                <div class='step-status-icon'>👁️</div>
                              </div>
                              <div class='step-score-row'>
                                <div>
                                  <span class='step-score-big notvisible'>0</span>
                                  <span class='step-score-suffix'>/100</span>
                                </div>
                                <div class='step-fail-rate'>Not visible</div>
                              </div>
                              <div class='step-image-wrap'>
                            """, unsafe_allow_html=True)

                            img_key = f"step_{step_num}"
                            img_path = step_imgs.get(img_key)
                            if img_path and os.path.exists(img_path):
                                st.image(img_path)

                            st.markdown(f"""
                              </div>
                              <div class='step-notvisible-msg'>
                                <strong>Cannot evaluate:</strong> {s.get('issue') or 'body part not visible in the frame'}
                              </div>
                              <div class='step-cue'>
                                <strong>Cue:</strong> {s['cue']}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            passed = s["passed_overall"]
                            pf_class = "passed" if passed else "failed"
                            score_class = "pass" if passed else "fail"
                            icon = "✅" if passed else "⚠️"

                            if mode == "photo":
                                rate_label = "Pass" if passed else "Needs work"
                            else:
                                rate_label = f"Failed in {s['fail_rate_percent']}% of frames"

                            st.markdown(f"""
                            <div class='step-card {pf_class}'>
                              <div class='step-header'>
                                <div class='step-name'>Step {step_num}: {s['name']}</div>
                                <div class='step-status-icon'>{icon}</div>
                              </div>
                              <div class='step-score-row'>
                                <div>
                                  <span class='step-score-big {score_class}'>{s['average_score']}</span>
                                  <span class='step-score-suffix'>/100</span>
                                </div>
                                <div class='step-fail-rate'>{rate_label}</div>
                              </div>
                              <div class='step-image-wrap'>
                            """, unsafe_allow_html=True)

                            img_key = f"step_{step_num}"
                            img_path = step_imgs.get(img_key)
                            if img_path and os.path.exists(img_path):
                                st.image(img_path)

                            st.markdown("</div>", unsafe_allow_html=True)

                            if s["issue"]:
                                st.markdown(
                                    f"<div class='step-issue'><strong>Issue:</strong> {s['issue']}</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    "<div class='step-passed-msg'><strong>✓</strong> Looks good for this step.</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                f"<div class='step-cue'><strong>Cue:</strong> {s['cue']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("## Coach's Feedback")
        st.caption("Generated based on your step-by-step report.")
        st.text_area(
            "Feedback",
            value=st.session_state.gemini_feedback or "No feedback generated.",
            height=320,
            label_visibility="collapsed",
        )

elif not st.session_state.video_path:
    st.info("👆 Upload a video or take a photo above, then click **Analyze Pose**.")
