import os
import tempfile
from datetime import datetime
import streamlit as st

from src.pose_analyzer import analyze_video
from src.feedback import get_gemini_feedback

st.set_page_config(page_title="Tadasana Pose Detection and Feedback", layout="wide")
st.title("Tadasana Pose Detection and Feedback")

# -----------------------------------------------------------------------------
# Output directories
# Render's filesystem is ephemeral. We use a writable directory either next
# to the app (local dev) or in the system temp folder (cloud).
# -----------------------------------------------------------------------------
def get_output_dir(subdir: str) -> str:
    base = os.environ.get("OUTPUT_DIR")
    if base:
        path = os.path.join(base, subdir)
    else:
        # try local "output/" first; fall back to temp
        local = os.path.join("output", subdir)
        try:
            os.makedirs(local, exist_ok=True)
            test = os.path.join(local, ".write_test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            path = local
        except OSError:
            path = os.path.join(tempfile.gettempdir(), "tadasana", subdir)
    os.makedirs(path, exist_ok=True)
    return path


recordings_dir = get_output_dir("recordings")
frames_dir = get_output_dir("extracted_frames")

# Session state
for key, default in [
    ("video_path", None),
    ("analysis_result", None),
    ("gemini_feedback", None),
    ("capture_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------------------------------------
# Upload / capture
# -----------------------------------------------------------------------------
st.write("### Step 1: Record or upload your Tadasana video")
uploaded_video = st.file_uploader(
    "Upload recorded Tadasana video",
    type=["mp4", "mov", "avi", "mkv"],
)

st.write("### Step 2: Or capture from camera")
camera_file = st.camera_input("Capture a pose snapshot")

if uploaded_video is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(recordings_dir, f"uploaded_{timestamp}.mp4")
    with open(save_path, "wb") as f:
        f.write(uploaded_video.read())
    st.session_state.video_path = save_path
    st.session_state.capture_done = True
    st.success("Video uploaded successfully.")

if camera_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(frames_dir, f"captured_pose_{timestamp}.jpg")
    with open(image_path, "wb") as f:
        f.write(camera_file.read())
    st.session_state.video_path = image_path
    st.session_state.capture_done = True
    st.success("Pose snapshot captured successfully.")

if st.session_state.video_path:
    st.write("### Captured Input")
    if st.session_state.video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        st.video(st.session_state.video_path)
    else:
        st.image(st.session_state.video_path, width="stretch")

analyze_btn = st.button("Analyze Pose")

# -----------------------------------------------------------------------------
# Run analysis
# -----------------------------------------------------------------------------
if analyze_btn and st.session_state.video_path:
    with st.spinner("Analyzing pose and generating feedback..."):
        try:
            if st.session_state.video_path.lower().endswith((".jpg", ".jpeg", ".png")):
                result = {
                    "final_score": 75,
                    "issues": ["Snapshot mode: upload a full video for accurate scoring"],
                    "steps": [],
                    "best_frame_path": st.session_state.video_path,
                    "annotated_path": None,
                    "step_image_paths": {},
                    "low_quality_warning": False,
                    "low_quality_message": None,
                }
            else:
                result = analyze_video(st.session_state.video_path, frames_dir)

            feedback_text = get_gemini_feedback(
                result["final_score"],
                result["issues"],
                steps=result.get("steps"),
            )
            st.session_state.analysis_result = result
            st.session_state.gemini_feedback = feedback_text

        except Exception as e:
            st.session_state.analysis_result = {
                "final_score": 0,
                "issues": [f"Analysis failed: {str(e)}"],
                "steps": [],
                "best_frame_path": None,
                "annotated_path": None,
                "step_image_paths": {},
                "low_quality_warning": True,
                "low_quality_message": f"Analysis failed: {str(e)}",
            }
            st.session_state.gemini_feedback = (
                f"Could not generate full pose feedback.\n\nReason: {str(e)}"
            )

# -----------------------------------------------------------------------------
# Display results
# -----------------------------------------------------------------------------
if st.session_state.analysis_result:
    result = st.session_state.analysis_result

    st.write("## Pose Analysis")

    if result.get("low_quality_warning"):
        st.warning(result.get("low_quality_message")
                   or "Low confidence in this analysis - please re-record.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.metric("Final Score", f"{result['final_score']}/100")
        st.write("### Detected Issues")
        if result["issues"]:
            for issue in result["issues"]:
                st.write(f"- {issue}")
        else:
            st.write("- No major issues detected")

    with col_right:
        annotated_path = result.get("annotated_path")
        if annotated_path and os.path.exists(annotated_path):
            st.write("### Annotated Best Frame")
            st.caption("Green = step passed - Red = step needs work")
            st.image(annotated_path, width="stretch")
        elif result.get("best_frame_path") and os.path.exists(result["best_frame_path"]):
            st.write("### Best Pose Frame")
            st.image(result["best_frame_path"], width="stretch")

    steps = result.get("steps") or []
    step_imgs = result.get("step_image_paths") or {}

    if steps:
        st.write("---")
        st.write("## Step-by-Step Breakdown")
        st.caption("Each step shows a zoomed crop of the body part being checked, "
                   "its score, and what to improve.")

        for i in range(0, len(steps), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(steps):
                    continue
                s = steps[i + j]
                step_num = s["step"]
                with col:
                    status_emoji = "✅" if s["passed_overall"] else "⚠️"
                    score_color = "green" if s["passed_overall"] else "red"
                    st.markdown(
                        f"### {status_emoji} Step {step_num}: {s['name']}"
                    )
                    st.markdown(
                        f"**Score: <span style='color:{score_color}'>"
                        f"{s['average_score']}/100</span>** "
                        f"&nbsp;&nbsp; Failed in {s['fail_rate_percent']}% of frames",
                        unsafe_allow_html=True,
                    )

                    img_key = f"step_{step_num}"
                    img_path = step_imgs.get(img_key)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width="stretch")

                    if s["issue"]:
                        st.error(f"**Issue:** {s['issue']}")
                    else:
                        st.success("**Looks good** for this step.")
                    st.info(f"**Cue:** {s['cue']}")
                    st.write("")

    st.write("---")
    st.write("## Final Feedback")
    st.text_area(
        "Feedback",
        value=st.session_state.gemini_feedback or "No feedback generated.",
        height=300,
    )

else:
    st.info("Upload a video or capture a snapshot, then click Analyze Pose.")