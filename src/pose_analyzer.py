"""
Pose analyzer for Utthita Balasana (Extended Child's Pose).

Two entry points:
  - analyze_video(path)  : process video frame by frame, aggregate scores
  - analyze_image(path)  : process a single photo

RULES:
  1. Visibility-zero rule: if a body part needed for a step isn't visible,
     that step scores 0 with a clear "not visible" message.
  2. Floor-pose check: if the user is detected as STANDING UP (torso vertical),
     refuse to score - return clear "this is not Child's Pose" message.
"""

import cv2
import os
import math
from src.pose_detector import PoseDetector
from src.scorer import calculate_angle, validate_pose

MIN_QUALITY_SCORE = 50
VISIBILITY_THRESHOLD = 0.5

# Torso must be tilted at least this many degrees from vertical to count as
# a floor pose. Less than this -> person is standing -> not Child's Pose.
MIN_TORSO_TILT_FOR_FLOOR_POSE = 30.0

POSE_LANDMARKS = {
    "nose": 0,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

STEP_CRITICAL_LANDMARKS = {
    1: ["left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel"],
    2: ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_knee", "right_knee"],
    3: ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"],
    4: ["left_hip", "right_hip", "left_shoulder", "right_shoulder",
        "left_wrist", "right_wrist"],
    5: ["nose", "left_wrist", "right_wrist"],
    6: ["left_shoulder", "right_shoulder", "left_ear", "right_ear"],
}


def extract_xy(landmarks, w, h, idx):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_from_vertical(p_top, p_bottom):
    """Angle of the line from p_bottom to p_top, measured from vertical.
    0 = perfectly vertical. 90 = perfectly horizontal."""
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


# -----------------------------------------------------------------------------
# Visibility check
# -----------------------------------------------------------------------------
def landmark_is_visible(lms, idx):
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESHOLD:
        return False
    if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
        return False
    return True


def get_step_visibility(lms):
    result = {}
    for step_num, names in STEP_CRITICAL_LANDMARKS.items():
        all_visible = True
        for name in names:
            idx = POSE_LANDMARKS[name]
            if not landmark_is_visible(lms, idx):
                all_visible = False
                break
        result[step_num] = all_visible
    return result


# -----------------------------------------------------------------------------
# NEW: Floor-pose check (rejects standing/upright poses)
# -----------------------------------------------------------------------------
def get_torso_tilt(lms, w, h):
    """Return the torso's angle from vertical, in degrees.
    0 = perfectly upright (standing). 90 = lying horizontal (folded forward).
    Returns None if shoulders or hips are not visible."""
    needed = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for name in needed:
        if not landmark_is_visible(lms, POSE_LANDMARKS[name]):
            return None
    ls = extract_xy(lms, w, h, POSE_LANDMARKS["left_shoulder"])
    rs = extract_xy(lms, w, h, POSE_LANDMARKS["right_shoulder"])
    lh = extract_xy(lms, w, h, POSE_LANDMARKS["left_hip"])
    rh = extract_xy(lms, w, h, POSE_LANDMARKS["right_hip"])
    mid_shoulders = midpoint(ls, rs)
    mid_hips = midpoint(lh, rh)
    return angle_from_vertical(mid_shoulders, mid_hips)


def is_floor_pose(lms, w, h):
    """Returns True if the body looks like it's in a folded/floor pose.
    Returns False if the body is standing upright (e.g. Tadasana)."""
    tilt = get_torso_tilt(lms, w, h)
    if tilt is None:
        return True  # if we can't tell, don't reject - let visibility rule handle it
    return tilt >= MIN_TORSO_TILT_FOR_FLOOR_POSE


# -----------------------------------------------------------------------------
# Feature extraction for Child's Pose
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
    lhe = extract_xy(lms, w, h, POSE_LANDMARKS["left_heel"])
    rhe = extract_xy(lms, w, h, POSE_LANDMARKS["right_heel"])
    le = extract_xy(lms, w, h, POSE_LANDMARKS["left_ear"])
    re = extract_xy(lms, w, h, POSE_LANDMARKS["right_ear"])
    nose = extract_xy(lms, w, h, POSE_LANDMARKS["nose"])

    mid_shoulders = midpoint(ls, rs)
    mid_hips = midpoint(lh, rh)
    mid_knees = midpoint(lk, rk)
    mid_heels = midpoint(lhe, rhe)
    mid_wrists = midpoint(lw_pt, rw_pt)
    mid_ears = midpoint(le, re)

    body_scale = distance(mid_shoulders, mid_hips)
    if body_scale < 1:
        body_scale = max(w, h) * 0.1

    thigh_length = distance(mid_hips, mid_knees)
    if thigh_length < 1:
        thigh_length = body_scale
    hip_to_heel_distance = distance(mid_hips, mid_heels)
    hip_to_heel_ratio = hip_to_heel_distance / thigh_length

    torso_thigh_angle = calculate_angle(mid_shoulders, mid_hips, mid_knees)

    left_elbow_angle = calculate_angle(ls, lel, lw_pt)
    right_elbow_angle = calculate_angle(rs, rel, rw_pt)

    left_se = distance(ls, lel)
    right_se = distance(rs, rel)
    left_sw = distance(ls, lw_pt)
    right_sw = distance(rs, rw_pt)
    left_arm_extension_ratio = left_sw / left_se if left_se > 1 else 1.0
    right_arm_extension_ratio = right_sw / right_se if right_se > 1 else 1.0

    spine_line_angle = calculate_angle(mid_hips, mid_shoulders, mid_wrists)
    spine_line_deviation = 180.0 - spine_line_angle

    head_lift_above_mat = (mid_wrists[1] - nose[1]) / body_scale
    shoulder_ear_drop = (mid_shoulders[1] - mid_ears[1]) / body_scale

    return {
        "hip_to_heel_ratio": hip_to_heel_ratio,
        "torso_thigh_angle": torso_thigh_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_arm_extension_ratio": left_arm_extension_ratio,
        "right_arm_extension_ratio": right_arm_extension_ratio,
        "spine_line_deviation": spine_line_deviation,
        "head_lift_above_mat": head_lift_above_mat,
        "shoulder_ear_drop": shoulder_ear_drop,
        "body_scale": body_scale,
    }


# -----------------------------------------------------------------------------
# Image generation (unchanged from previous version)
# -----------------------------------------------------------------------------
def _crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


def _crop_with_padding(img, points, padding_x_frac=0.15, padding_y_frac=0.15):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h, w = img.shape[:2]
    pad_x = w * padding_x_frac
    pad_y = h * padding_y_frac
    x1 = min(xs) - pad_x; x2 = max(xs) + pad_x
    y1 = min(ys) - pad_y; y2 = max(ys) + pad_y
    return _crop_safe(img, x1, y1, x2, y2)


def generate_step_images(frame, lms, step_results, save_dir):
    h, w = frame.shape[:2]
    paths = {}

    pts = {name: extract_xy(lms, w, h, idx)
           for name, idx in POSE_LANDMARKS.items()}

    def step_state(step_num):
        for s in step_results:
            if s["step"] == step_num:
                if s.get("not_visible"):
                    return "not_visible"
                return "passed" if s["passed"] else "failed"
        return "failed"

    annotated = frame.copy()
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    GRAY = (130, 130, 130)

    def color_for(step_num):
        st = step_state(step_num)
        if st == "passed": return GREEN
        if st == "not_visible": return GRAY
        return RED

    def line(p1, p2, color, thick=4):
        cv2.line(annotated,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 color, thick, cv2.LINE_AA)

    def dot(p, color, r=6):
        cv2.circle(annotated, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)

    c1 = color_for(1)
    mid_hp = midpoint(pts["left_hip"], pts["right_hip"])
    mid_he = midpoint(pts["left_heel"], pts["right_heel"])
    line(mid_hp, mid_he, c1, thick=3)

    c2 = color_for(2)
    mid_sh = midpoint(pts["left_shoulder"], pts["right_shoulder"])
    mid_kn = midpoint(pts["left_knee"], pts["right_knee"])
    line(mid_sh, mid_hp, c2, thick=5)
    line(mid_hp, mid_kn, c2, thick=5)

    c3 = color_for(3)
    line(pts["left_shoulder"], pts["left_elbow"], c3)
    line(pts["left_elbow"], pts["left_wrist"], c3)
    line(pts["right_shoulder"], pts["right_elbow"], c3)
    line(pts["right_elbow"], pts["right_wrist"], c3)

    c4 = color_for(4)
    mid_wr = midpoint(pts["left_wrist"], pts["right_wrist"])
    line(mid_hp, mid_sh, c4, thick=4)
    line(mid_sh, mid_wr, c4, thick=4)

    c5 = color_for(5)
    dot(pts["nose"], c5, r=10)

    c6 = color_for(6)
    line(pts["left_ear"], pts["left_shoulder"], c6, thick=3)
    line(pts["right_ear"], pts["right_shoulder"], c6, thick=3)

    for name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle",
                 "left_heel", "right_heel"]:
        dot(pts[name], (255, 255, 255), r=4)

    annotated_path = os.path.join(save_dir, "annotated_full.jpg")
    cv2.imwrite(annotated_path, annotated)
    paths["annotated"] = annotated_path

    hh_pts = [pts["left_hip"], pts["right_hip"],
              pts["left_knee"], pts["right_knee"],
              pts["left_heel"], pts["right_heel"]]
    crop = _crop_with_padding(annotated, hh_pts, 0.12, 0.12)
    p1 = os.path.join(save_dir, "step1_hips_on_heels.jpg")
    cv2.imwrite(p1, crop); paths["step_1"] = p1

    tf_pts = [pts["left_shoulder"], pts["right_shoulder"],
              pts["left_hip"], pts["right_hip"],
              pts["left_knee"], pts["right_knee"]]
    crop = _crop_with_padding(annotated, tf_pts, 0.10, 0.10)
    p2 = os.path.join(save_dir, "step2_torso_fold.jpg")
    cv2.imwrite(p2, crop); paths["step_2"] = p2

    arm_pts = [pts["left_shoulder"], pts["right_shoulder"],
               pts["left_elbow"], pts["right_elbow"],
               pts["left_wrist"], pts["right_wrist"]]
    crop = _crop_with_padding(annotated, arm_pts, 0.10, 0.15)
    p3 = os.path.join(save_dir, "step3_arms_extended.jpg")
    cv2.imwrite(p3, crop); paths["step_3"] = p3

    sp_pts = [pts["left_hip"], pts["right_hip"],
              pts["left_shoulder"], pts["right_shoulder"],
              pts["left_wrist"], pts["right_wrist"]]
    crop = _crop_with_padding(annotated, sp_pts, 0.10, 0.10)
    p4 = os.path.join(save_dir, "step4_spine_lengthened.jpg")
    cv2.imwrite(p4, crop); paths["step_4"] = p4

    head_pts = [pts["nose"], pts["left_wrist"], pts["right_wrist"]]
    crop = _crop_with_padding(annotated, head_pts, 0.15, 0.20)
    p5 = os.path.join(save_dir, "step5_forehead_down.jpg")
    cv2.imwrite(p5, crop); paths["step_5"] = p5

    sh_pts = [pts["left_shoulder"], pts["right_shoulder"],
              pts["left_ear"], pts["right_ear"]]
    crop = _crop_with_padding(annotated, sh_pts, 0.15, 0.20)
    p6 = os.path.join(save_dir, "step6_shoulders_relaxed.jpg")
    cv2.imwrite(p6, crop); paths["step_6"] = p6

    return paths


# -----------------------------------------------------------------------------
# NEW: Build a "not Child's Pose" result for standing/wrong-pose inputs
# -----------------------------------------------------------------------------
def _not_childs_pose_result(best_frame_path=None, source_image_path=None,
                            annotated_path=None, mode="video"):
    """Return a result dict indicating the input is NOT Child's Pose."""
    return {
        "final_score": 0,
        "issues": ["This does not look like Extended Child's Pose - "
                   "you appear to be standing or in a different pose"],
        "steps": [],
        "best_frame_path": best_frame_path or source_image_path,
        "annotated_path": annotated_path,
        "step_image_paths": {},
        "low_quality_warning": True,
        "low_quality_message": (
            "We did not detect Extended Child's Pose. "
            "Please get into the pose (kneeling, forehead toward the mat, "
            "arms extended forward) and re-record from the SIDE of your body."
        ),
        "pose_invalid": True,
    }


# -----------------------------------------------------------------------------
# Aggregation across frames
# -----------------------------------------------------------------------------
def aggregate_step_reports(all_reports):
    if not all_reports:
        return None

    num_steps = len(all_reports[0]["steps"])
    aggregated_steps = []

    for i in range(num_steps):
        scores = []
        issues_seen = []
        fails = 0
        not_visible_count = 0
        cue = ""; name = ""; weight = 0
        for report in all_reports:
            s = report["steps"][i]
            scores.append(s["score"])
            cue = s["cue"]; name = s["name"]; weight = s["weight"]
            if not s["passed"]:
                fails += 1
            if s.get("not_visible"):
                not_visible_count += 1
            if s["issue"]:
                issues_seen.append(s["issue"])

        avg = round(sum(scores) / len(scores), 1)
        fail_rate = round(fails / len(all_reports) * 100, 1)
        not_visible_rate = round(not_visible_count / len(all_reports) * 100, 1)
        most_common = max(set(issues_seen), key=issues_seen.count) if issues_seen else None
        not_visible_overall = not_visible_rate > 50

        aggregated_steps.append({
            "step": i + 1,
            "name": name, "cue": cue, "weight": weight,
            "average_score": avg,
            "fail_rate_percent": fail_rate,
            "not_visible_rate_percent": not_visible_rate,
            "not_visible": not_visible_overall,
            "issue": most_common,
            "passed_overall": fail_rate < 25 and not not_visible_overall,
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


def _single_frame_to_aggregated(report):
    aggregated_steps = []
    for s in report["steps"]:
        not_vis = s.get("not_visible", False)
        aggregated_steps.append({
            "step": s["step"],
            "name": s["name"],
            "cue": s["cue"],
            "weight": s["weight"],
            "average_score": round(s["score"], 1),
            "fail_rate_percent": 0.0 if s["passed"] else 100.0,
            "not_visible_rate_percent": 100.0 if not_vis else 0.0,
            "not_visible": not_vis,
            "issue": s["issue"],
            "passed_overall": s["passed"] and not not_vis,
        })
    return {
        "final_score": report["final_score"],
        "steps": aggregated_steps,
        "issues": report["issues"],
    }


# -----------------------------------------------------------------------------
# Video analysis
# -----------------------------------------------------------------------------
def analyze_video(video_path, save_frames_dir=None):
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)

    all_reports = []
    best_score = -1
    best_frame = None
    best_landmarks = None
    best_step_results = None

    floor_pose_frames = 0
    standing_pose_frames = 0

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

            # Floor-pose check
            if is_floor_pose(lms, w, h):
                floor_pose_frames += 1
            else:
                standing_pose_frames += 1
                continue  # skip this frame for scoring - person is standing

            step_visibility = get_step_visibility(lms)
            features = build_features(lms, w, h)
            report = validate_pose(features, step_visibility)
            all_reports.append(report)

            if report["final_score"] > best_score:
                best_score = report["final_score"]
                best_frame = frame.copy()
                best_landmarks = lms
                best_step_results = report["steps"]

    cap.release()

    total_pose_frames = floor_pose_frames + standing_pose_frames

    # If MOST frames showed standing pose, reject as not Child's Pose
    if total_pose_frames > 0 and standing_pose_frames > floor_pose_frames:
        return _not_childs_pose_result(mode="video")

    if not all_reports:
        return {
            "final_score": 0,
            "issues": ["No pose detected in the video"],
            "steps": [],
            "best_frame_path": None,
            "annotated_path": None,
            "step_image_paths": {},
            "low_quality_warning": True,
            "low_quality_message": (
                "No body pose detected. Please record from the SIDE "
                "with your full body in frame."
            ),
        }

    aggregated = aggregate_step_reports(all_reports)

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

    low_quality = best_score < MIN_QUALITY_SCORE
    low_quality_msg = None
    if low_quality:
        low_quality_msg = (
            f"The best frame scored only {best_score}/100. "
            "For accurate results: record from the SIDE of your body, "
            "with good lighting and the pose held steadily."
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


# -----------------------------------------------------------------------------
# Photo analysis
# -----------------------------------------------------------------------------
def analyze_image(image_path, save_frames_dir=None):
    detector = PoseDetector()
    frame = cv2.imread(image_path)

    if frame is None:
        return {
            "final_score": 0,
            "issues": ["Could not read the uploaded image"],
            "steps": [],
            "best_frame_path": None,
            "annotated_path": None,
            "step_image_paths": {},
            "low_quality_warning": True,
            "low_quality_message": "The image file could not be read.",
        }

    h, w = frame.shape[:2]
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    results = detector.detect(frame)

    if not results.pose_landmarks:
        return {
            "final_score": 0,
            "issues": ["No body pose detected in the photo"],
            "steps": [],
            "best_frame_path": image_path,
            "annotated_path": None,
            "step_image_paths": {},
            "low_quality_warning": True,
            "low_quality_message": (
                "No body pose was detected. Please retake with: good lighting, "
                "SIDE view of your body, and clear contrast against the background."
            ),
        }

    lms = results.pose_landmarks.landmark

    # Floor-pose check FIRST
    if not is_floor_pose(lms, w, h):
        return _not_childs_pose_result(source_image_path=image_path, mode="photo")

    step_visibility = get_step_visibility(lms)
    features = build_features(lms, w, h)
    report = validate_pose(features, step_visibility)

    step_image_paths = {}
    annotated_path = None
    best_frame_path = None
    if save_frames_dir:
        best_frame_path = os.path.join(save_frames_dir, "best_pose_frame.jpg")
        cv2.imwrite(best_frame_path, frame.copy())
        step_image_paths = generate_step_images(
            frame, lms, report["steps"], save_frames_dir
        )
        annotated_path = step_image_paths.get("annotated")

    aggregated = _single_frame_to_aggregated(report)

    low_quality = aggregated["final_score"] < MIN_QUALITY_SCORE
    low_quality_msg = None
    if low_quality:
        low_quality_msg = (
            f"This photo scored only {aggregated['final_score']}/100. "
            "For accurate results: take the photo from the SIDE of your body, "
            "with good lighting and the pose held clearly."
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
