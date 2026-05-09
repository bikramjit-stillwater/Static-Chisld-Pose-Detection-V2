"""
Pose analyzer for arms-overhead Tadasana.

Per analysed video:
  - runs MediaPipe on every frame
  - builds 14 features (incl. arm_drop, elbow angles, arm closeness, symmetry)
  - validates using scorer.validate_tadasana
  - picks the best frame (only if score >= MIN_QUALITY_SCORE, else flags low quality)
  - generates 7 output images:
      * annotated_full.jpg          (skeleton overlay, green=pass, red=fail)
      * step1_stance.jpg            (cropped feet)
      * step2_body_balance.jpg      (full body)
      * step3_legs_knees.jpg        (cropped legs)
      * step4_spine.jpg             (cropped torso)
      * step5_shoulders_arms.jpg    (cropped upper body + raised arms)
      * step6_head_neck.jpg         (cropped head)
"""

import cv2
import os
import math
from src.pose_detector import PoseDetector
from src.scorer import calculate_angle, validate_tadasana

# Below this best-frame score we flag the analysis as low-confidence
MIN_QUALITY_SCORE = 50

POSE_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


def extract_xy(landmarks, w, h, idx):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def angle_from_vertical(p_top, p_bottom):
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------
def build_features(lms, w, h):
    ls = extract_xy(lms, w, h, POSE_LANDMARKS["left_shoulder"])
    rs = extract_xy(lms, w, h, POSE_LANDMARKS["right_shoulder"])
    lh = extract_xy(lms, w, h, POSE_LANDMARKS["left_hip"])
    rh = extract_xy(lms, w, h, POSE_LANDMARKS["right_hip"])
    lel = extract_xy(lms, w, h, POSE_LANDMARKS["left_elbow"])
    rel = extract_xy(lms, w, h, POSE_LANDMARKS["right_elbow"])
    lw_pt = extract_xy(lms, w, h, POSE_LANDMARKS["left_wrist"])
    rw_pt = extract_xy(lms, w, h, POSE_LANDMARKS["right_wrist"])
    lk = extract_xy(lms, w, h, POSE_LANDMARKS["left_knee"])
    rk = extract_xy(lms, w, h, POSE_LANDMARKS["right_knee"])
    la = extract_xy(lms, w, h, POSE_LANDMARKS["left_ankle"])
    ra = extract_xy(lms, w, h, POSE_LANDMARKS["right_ankle"])
    nose = extract_xy(lms, w, h, POSE_LANDMARKS["nose"])

    # Tilts
    shoulder_tilt = abs(ls[1] - rs[1]) / h
    hip_tilt = abs(lh[1] - rh[1]) / h

    # Body lean
    body_center_x = ((ls[0] + rs[0]) / 2 + (lh[0] + rh[0]) / 2) / 2
    ankle_center_x = (la[0] + ra[0]) / 2
    body_lean = abs(body_center_x - ankle_center_x) / w

    # Knee angles
    left_knee_bend = calculate_angle(lh, lk, la)
    right_knee_bend = calculate_angle(rh, rk, ra)

    # Stance ratio
    ankle_distance = abs(la[0] - ra[0])
    hip_distance = abs(lh[0] - rh[0])
    stance_ratio = ankle_distance / hip_distance if hip_distance > 1 else 1.0

    # Spine tilt
    mid_shoulders = midpoint(ls, rs)
    mid_hips = midpoint(lh, rh)
    spine_tilt = angle_from_vertical(mid_shoulders, mid_hips)

    # Head offset
    head_offset = abs(nose[0] - mid_shoulders[0]) / w

    # Arm vertical drop (negative = wrist above shoulder = arms raised)
    shoulder_y = (ls[1] + rs[1]) / 2
    ankle_y = (la[1] + ra[1]) / 2
    body_height = ankle_y - shoulder_y
    if body_height > 1:
        left_arm_drop = (lw_pt[1] - shoulder_y) / body_height
        right_arm_drop = (rw_pt[1] - shoulder_y) / body_height
    else:
        left_arm_drop = 0.5
        right_arm_drop = 0.5

    # Elbow angles (shoulder-elbow-wrist)
    left_elbow_angle = calculate_angle(ls, lel, lw_pt)
    right_elbow_angle = calculate_angle(rs, rel, rw_pt)

    # Arm closeness (horizontal distance between wrists, normalized by image width)
    arm_closeness = abs(lw_pt[0] - rw_pt[0]) / w

    return {
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "body_lean": body_lean,
        "left_knee_bend": left_knee_bend,
        "right_knee_bend": right_knee_bend,
        "stance_ratio": stance_ratio,
        "spine_tilt": spine_tilt,
        "head_offset": head_offset,
        "left_arm_drop": left_arm_drop,
        "right_arm_drop": right_arm_drop,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "arm_closeness": arm_closeness,
    }


# -----------------------------------------------------------------------------
# Image generation - crops + annotated full image
# -----------------------------------------------------------------------------
def _crop_safe(img, x1, y1, x2, y2):
    """Crop img[y1:y2, x1:x2] safely, clamping to image bounds."""
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


def _crop_with_padding(img, points, padding_x_frac=0.15, padding_y_frac=0.15):
    """Crop a bounding box around given (x,y) points with padding."""
    h, w = img.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad_x = w * padding_x_frac
    pad_y = h * padding_y_frac
    x1 = min(xs) - pad_x; x2 = max(xs) + pad_x
    y1 = min(ys) - pad_y; y2 = max(ys) + pad_y
    return _crop_safe(img, x1, y1, x2, y2)


def generate_step_images(frame, lms, step_results, save_dir):
    """
    Generate 6 cropped images (one per step) + 1 annotated full image.
    Returns dict: { 'annotated': path, 'step_1': path, ..., 'step_6': path }
    """
    h, w = frame.shape[:2]
    paths = {}

    # Get pixel positions of every relevant landmark
    pts = {name: extract_xy(lms, w, h, idx)
           for name, idx in POSE_LANDMARKS.items()}

    # Step -> pass/fail map
    step_passed = {s["step"]: s["passed"] for s in step_results}

    # ----- Annotated full image -----
    annotated = frame.copy()
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    GREY = (180, 180, 180)

    def line(p1, p2, color, thick=4):
        cv2.line(annotated,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 color, thick, cv2.LINE_AA)

    def dot(p, color, r=6):
        cv2.circle(annotated, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)

    # Step 5 (Shoulders & Arms) - shoulder-elbow-wrist lines
    c5 = GREEN if step_passed.get(5) else RED
    line(pts["left_shoulder"], pts["left_elbow"], c5)
    line(pts["left_elbow"], pts["left_wrist"], c5)
    line(pts["right_shoulder"], pts["right_elbow"], c5)
    line(pts["right_elbow"], pts["right_wrist"], c5)
    line(pts["left_shoulder"], pts["right_shoulder"], c5)

    # Step 4 (Spine) - shoulder-hip line
    c4 = GREEN if step_passed.get(4) else RED
    mid_sh = midpoint(pts["left_shoulder"], pts["right_shoulder"])
    mid_hp = midpoint(pts["left_hip"], pts["right_hip"])
    line(mid_sh, mid_hp, c4, thick=5)

    # Step 3 (Legs & Knees) - hip-knee-ankle
    c3 = GREEN if step_passed.get(3) else RED
    line(pts["left_hip"], pts["left_knee"], c3)
    line(pts["left_knee"], pts["left_ankle"], c3)
    line(pts["right_hip"], pts["right_knee"], c3)
    line(pts["right_knee"], pts["right_ankle"], c3)
    line(pts["left_hip"], pts["right_hip"], c3)

    # Step 1 (Stance) - foot-to-foot line
    c1 = GREEN if step_passed.get(1) else RED
    line(pts["left_ankle"], pts["right_ankle"], c1, thick=3)

    # Step 6 (Head) - dot on nose
    c6 = GREEN if step_passed.get(6) else RED
    dot(pts["nose"], c6, r=10)

    # All joint dots
    for name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"]:
        dot(pts[name], (255, 255, 255), r=4)

    annotated_path = os.path.join(save_dir, "annotated_full.jpg")
    cv2.imwrite(annotated_path, annotated)
    paths["annotated"] = annotated_path

    # ----- Step 1: Stance (feet) -----
    feet_pts = [pts["left_ankle"], pts["right_ankle"],
                pts["left_heel"], pts["right_heel"],
                pts["left_foot_index"], pts["right_foot_index"]]
    crop = _crop_with_padding(annotated, feet_pts, 0.15, 0.10)
    p1 = os.path.join(save_dir, "step1_stance.jpg")
    cv2.imwrite(p1, crop); paths["step_1"] = p1

    # ----- Step 2: Body Balance (full body) -----
    p2 = os.path.join(save_dir, "step2_body_balance.jpg")
    cv2.imwrite(p2, annotated); paths["step_2"] = p2

    # ----- Step 3: Legs & Knees (hips down to ankles) -----
    leg_pts = [pts["left_hip"], pts["right_hip"],
               pts["left_knee"], pts["right_knee"],
               pts["left_ankle"], pts["right_ankle"]]
    crop = _crop_with_padding(annotated, leg_pts, 0.12, 0.05)
    p3 = os.path.join(save_dir, "step3_legs_knees.jpg")
    cv2.imwrite(p3, crop); paths["step_3"] = p3

    # ----- Step 4: Spine (shoulders down to hips) -----
    spine_pts = [pts["left_shoulder"], pts["right_shoulder"],
                 pts["left_hip"], pts["right_hip"]]
    crop = _crop_with_padding(annotated, spine_pts, 0.18, 0.05)
    p4 = os.path.join(save_dir, "step4_spine.jpg")
    cv2.imwrite(p4, crop); paths["step_4"] = p4

    # ----- Step 5: Shoulders & Arms (need to include raised arms) -----
    arm_pts = [pts["left_shoulder"], pts["right_shoulder"],
               pts["left_elbow"], pts["right_elbow"],
               pts["left_wrist"], pts["right_wrist"],
               pts["left_hip"], pts["right_hip"]]
    crop = _crop_with_padding(annotated, arm_pts, 0.10, 0.10)
    p5 = os.path.join(save_dir, "step5_shoulders_arms.jpg")
    cv2.imwrite(p5, crop); paths["step_5"] = p5

    # ----- Step 6: Head & Neck -----
    head_pts = [pts["nose"], pts["left_shoulder"], pts["right_shoulder"]]
    crop = _crop_with_padding(annotated, head_pts, 0.15, 0.20)
    p6 = os.path.join(save_dir, "step6_head_neck.jpg")
    cv2.imwrite(p6, crop); paths["step_6"] = p6

    return paths


# -----------------------------------------------------------------------------
# Aggregation across frames
# -----------------------------------------------------------------------------
def aggregate_step_reports(all_reports):
    if not all_reports:
        return None

    num_steps = len(all_reports[0]["steps"])
    aggregated_steps = []

    for i in range(num_steps):
        scores = []; issues_seen = []; fails = 0
        cue = ""; name = ""; weight = 0
        for report in all_reports:
            s = report["steps"][i]
            scores.append(s["score"])
            cue = s["cue"]; name = s["name"]; weight = s["weight"]
            if not s["passed"]:
                fails += 1
            if s["issue"]:
                issues_seen.append(s["issue"])

        avg = round(sum(scores) / len(scores), 1)
        fail_rate = round(fails / len(all_reports) * 100, 1)
        most_common = max(set(issues_seen), key=issues_seen.count) if issues_seen else None

        aggregated_steps.append({
            "step": i + 1,
            "name": name, "cue": cue, "weight": weight,
            "average_score": avg,
            "fail_rate_percent": fail_rate,
            "issue": most_common,
            "passed_overall": fail_rate < 25,
        })

    finals = [r["final_score"] for r in all_reports]
    final_score = int(round(sum(finals) / len(finals)))
    final_score = max(0, min(100, final_score))

    significant_issues = [
        s["issue"] for s in aggregated_steps
        if s["issue"] and s["fail_rate_percent"] >= 25
    ]
    return {
        "final_score": final_score,
        "steps": aggregated_steps,
        "issues": significant_issues,
    }


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def analyze_video(video_path, save_frames_dir=None):
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)

    all_reports = []
    best_score = -1
    best_frame = None
    best_landmarks = None
    best_step_results = None

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        results = detector.detect(frame)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            features = build_features(lms, w, h)
            report = validate_tadasana(features)
            all_reports.append(report)

            if report["final_score"] > best_score:
                best_score = report["final_score"]
                best_frame = frame.copy()
                best_landmarks = lms
                best_step_results = report["steps"]

    cap.release()

    if not all_reports:
        return {
            "final_score": 0,
            "issues": ["No pose detected in the video"],
            "steps": [],
            "best_frame_path": None,
            "annotated_path": None,
            "step_image_paths": {},
            "low_quality_warning": True,
            "low_quality_message": "No body pose detected. Please record again with the full body in frame.",
        }

    aggregated = aggregate_step_reports(all_reports)

    # Generate the 7 images from the best frame
    step_image_paths = {}
    annotated_path = None
    best_frame_path = None
    if best_frame is not None and save_frames_dir:
        best_frame_path = os.path.join(save_frames_dir, "best_pose_frame.jpg")
        cv2.imwrite(best_frame_path, best_frame)
        step_image_paths = generate_step_images(
            best_frame, best_landmarks, best_step_results, save_frames_dir
        )
        annotated_path = step_image_paths.get("annotated")

    # Quality warning if best frame was not good enough
    low_quality = best_score < MIN_QUALITY_SCORE
    low_quality_msg = None
    if low_quality:
        low_quality_msg = (
            f"The best frame in this video only scored {best_score}/100 "
            f"(below the {MIN_QUALITY_SCORE} confidence threshold). "
            "The pose may not have been clearly Tadasana. "
            "For more accurate results, please re-record with: full body in frame, "
            "good lighting, and hold the pose steadily for a few seconds."
        )

    return {
        "final_score": aggregated["final_score"],
        "issues": aggregated["issues"],
        "steps": aggregated["steps"],
        "best_frame_path": best_frame_path,
        "annotated_path": annotated_path,
        "step_image_paths": step_image_paths,
        "low_quality_warning": low_quality,
        "low_quality_message": low_quality_msg,
    }