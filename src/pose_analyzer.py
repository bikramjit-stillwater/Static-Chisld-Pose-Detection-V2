"""
Pose analyzer for Child's Pose.

Two entry points:
  - analyze_video(path)  : process video frame by frame, aggregate scores
  - analyze_image(path)  : process a single photo

RULES:
  1. ONE-SIDE VISIBILITY: Child's Pose is recorded from the side, so only one
     side of the body faces the camera (the other side is naturally hidden).
     A step is considered VISIBLE if ONE FULL SIDE of its required body parts
     is visible - it does NOT require BOTH sides.
     If neither side is visible, the step scores 0 with "not visible" message.

  2. FLOOR-POSE CHECK: If torso is upright (standing), refuse to score
     with clear "this is not Child's Pose" message.

  3. SINGLE-SIDE FEATURES: Geometry is computed from the more-visible side
     of the body, never from midpoints between visible and invisible points.
"""

import cv2
import os
import math
from src.pose_detector import PoseDetector
from src.scorer import calculate_angle, validate_pose

MIN_QUALITY_SCORE = 50
VISIBILITY_THRESHOLD = 0.5
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

# Each step's critical landmarks are now ALTERNATIVE SETS.
# A step is "visible" if ANY ONE of these sets is fully visible.
# This is the key fix: side-view recordings only show ONE side of the body,
# and that's enough to evaluate the pose.
STEP_CRITICAL_LANDMARKS = {
    1: [  # Hips on Heels - need one full leg
        ["left_hip", "left_knee", "left_ankle", "left_heel"],
        ["right_hip", "right_knee", "right_ankle", "right_heel"],
    ],
    2: [  # Torso Folded Forward
        ["left_shoulder", "left_hip", "left_knee"],
        ["right_shoulder", "right_hip", "right_knee"],
    ],
    3: [  # Arms Extended Forward
        ["left_shoulder", "left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow", "right_wrist"],
    ],
    4: [  # Spine Lengthened
        ["left_hip", "left_shoulder", "left_wrist"],
        ["right_hip", "right_shoulder", "right_wrist"],
    ],
    5: [  # Forehead Down
        ["nose", "left_wrist"],
        ["nose", "right_wrist"],
    ],
    6: [  # Shoulders Relaxed
        ["left_shoulder", "left_ear"],
        ["right_shoulder", "right_ear"],
    ],
}


def extract_xy(landmarks, w, h, idx):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_from_vertical(p_top, p_bottom):
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


# -----------------------------------------------------------------------------
# Visibility check (NEW: any one set must be fully visible)
# -----------------------------------------------------------------------------
def landmark_is_visible(lms, idx):
    """A landmark is visible if MediaPipe confidence is high AND coords inside image."""
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESHOLD:
        return False
    if lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0:
        return False
    return True


def get_step_visibility(lms):
    """Return {step_num: bool}. A step is visible if ANY of its alternative
    landmark sets is fully visible (one full side is enough)."""
    result = {}
    for step_num, alternative_sets in STEP_CRITICAL_LANDMARKS.items():
        any_set_fully_visible = False
        for landmark_set in alternative_sets:
            all_in_set_visible = all(
                landmark_is_visible(lms, POSE_LANDMARKS[name])
                for name in landmark_set
            )
            if all_in_set_visible:
                any_set_fully_visible = True
                break
        result[step_num] = any_set_fully_visible
    return result


# -----------------------------------------------------------------------------
# Determine which side of the body is more visible (for feature computation)
# -----------------------------------------------------------------------------
def determine_visible_side(lms):
    """Compute which side of the body is more visible overall.
    Returns 'left' or 'right' - whichever side has higher cumulative
    visibility across key landmarks."""
    key_left = ["left_shoulder", "left_hip", "left_knee", "left_ankle",
                "left_elbow", "left_wrist", "left_heel", "left_ear"]
    key_right = ["right_shoulder", "right_hip", "right_knee", "right_ankle",
                 "right_elbow", "right_wrist", "right_heel", "right_ear"]

    left_score = 0.0
    right_score = 0.0
    for name in key_left:
        lm = lms[POSE_LANDMARKS[name]]
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            left_score += lm.visibility
    for name in key_right:
        lm = lms[POSE_LANDMARKS[name]]
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            right_score += lm.visibility

    return "left" if left_score >= right_score else "right"


# -----------------------------------------------------------------------------
# Floor-pose check (rejects standing/upright poses)
# -----------------------------------------------------------------------------
def get_torso_tilt(lms, w, h, side):
    """Compute torso tilt from vertical using ONE side of the body."""
    sh_name = f"{side}_shoulder"
    hp_name = f"{side}_hip"
    if not landmark_is_visible(lms, POSE_LANDMARKS[sh_name]):
        return None
    if not landmark_is_visible(lms, POSE_LANDMARKS[hp_name]):
        return None
    shoulder = extract_xy(lms, w, h, POSE_LANDMARKS[sh_name])
    hip = extract_xy(lms, w, h, POSE_LANDMARKS[hp_name])
    return angle_from_vertical(shoulder, hip)


def is_floor_pose(lms, w, h):
    """Returns True if the body is in a folded/floor pose (not standing)."""
    side = determine_visible_side(lms)
    tilt = get_torso_tilt(lms, w, h, side)
    if tilt is None:
        # Try the other side
        other = "right" if side == "left" else "left"
        tilt = get_torso_tilt(lms, w, h, other)
    if tilt is None:
        return True  # don't reject if we can't measure
    return tilt >= MIN_TORSO_TILT_FOR_FLOOR_POSE


# -----------------------------------------------------------------------------
# Feature extraction - uses ONE SIDE of the body (no more midpoints)
# -----------------------------------------------------------------------------
def build_features(lms, w, h):
    """Build features using the more-visible side of the body.
    Never combines left and right (no midpoints), so we never use
    extrapolated/invisible landmark positions."""

    side = determine_visible_side(lms)

    def get(part):
        """Get the xy of part from the chosen side. For 'nose', always use nose."""
        if part == "nose":
            return extract_xy(lms, w, h, POSE_LANDMARKS["nose"])
        return extract_xy(lms, w, h, POSE_LANDMARKS[f"{side}_{part}"])

    shoulder = get("shoulder")
    hip = get("hip")
    knee = get("knee")
    ankle = get("ankle")
    heel = get("heel")
    elbow = get("elbow")
    wrist = get("wrist")
    ear = get("ear")
    nose = get("nose")

    # body_scale: shoulder-to-hip distance (this is most stable for any pose)
    body_scale = distance(shoulder, hip)
    if body_scale < 1:
        body_scale = max(w, h) * 0.1

    # Step 1 features: Hips on Heels
    thigh_length = distance(hip, knee)
    if thigh_length < 1:
        thigh_length = body_scale
    hip_to_heel_distance = distance(hip, heel)
    hip_to_heel_ratio = hip_to_heel_distance / thigh_length

    # Step 2: Torso angle relative to thigh
    torso_thigh_angle = calculate_angle(shoulder, hip, knee)

    # Step 3: Arm features (using the visible side)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    shoulder_to_elbow = distance(shoulder, elbow)
    shoulder_to_wrist = distance(shoulder, wrist)
    arm_extension_ratio = (shoulder_to_wrist / shoulder_to_elbow
                           if shoulder_to_elbow > 1 else 1.0)

    # Step 4: Spine line (hip -> shoulder -> wrist should be near straight)
    spine_line_angle = calculate_angle(hip, shoulder, wrist)
    spine_line_deviation = 180.0 - spine_line_angle

    # Step 5: How much nose is above wrist level
    head_lift_above_mat = (wrist[1] - nose[1]) / body_scale

    # Step 6: Shoulder below ear
    shoulder_ear_drop = (shoulder[1] - ear[1]) / body_scale

    # Scorer expects both left and right elbow/arm values - we provide
    # the visible-side value for both, and 0 asymmetry (since we can't
    # measure asymmetry with only one side visible - we assume symmetric).
    return {
        "hip_to_heel_ratio": hip_to_heel_ratio,
        "torso_thigh_angle": torso_thigh_angle,
        "left_elbow_angle": elbow_angle,
        "right_elbow_angle": elbow_angle,
        "left_arm_extension_ratio": arm_extension_ratio,
        "right_arm_extension_ratio": arm_extension_ratio,
        "spine_line_deviation": spine_line_deviation,
        "head_lift_above_mat": head_lift_above_mat,
        "shoulder_ear_drop": shoulder_ear_drop,
        "body_scale": body_scale,
        "_visible_side": side,
    }


# -----------------------------------------------------------------------------
# Image generation - only draws visible landmarks
# -----------------------------------------------------------------------------
def _crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


def _crop_with_padding(img, points, padding_x_frac=0.15, padding_y_frac=0.15):
    if not points:
        return img.copy()
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

    side = determine_visible_side(lms)

    def get_pt(name):
        return extract_xy(lms, w, h, POSE_LANDMARKS[name])

    def is_vis(name):
        return landmark_is_visible(lms, POSE_LANDMARKS[name])

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

    def color_for(step_num):
        st = step_state(step_num)
        if st == "passed": return GREEN
        if st == "not_visible": return GREEN  # don't draw not-visible at all
        return RED

    def line(p1, p2, color, thick=4):
        cv2.line(annotated,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 color, thick, cv2.LINE_AA)

    def dot(p, color, r=6):
        cv2.circle(annotated, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)

    # Determine which side to draw based on visibility
    def draw_side_line(part_a, part_b, color, thick=4):
        """Draw line between two body parts on the visible side, ONLY IF
        both parts on that side are visible."""
        a_name = f"{side}_{part_a}"
        b_name = f"{side}_{part_b}"
        if is_vis(a_name) and is_vis(b_name):
            line(get_pt(a_name), get_pt(b_name), color, thick)
        else:
            # Try other side as fallback
            other = "right" if side == "left" else "left"
            a_name = f"{other}_{part_a}"
            b_name = f"{other}_{part_b}"
            if is_vis(a_name) and is_vis(b_name):
                line(get_pt(a_name), get_pt(b_name), color, thick)

    # Step 1 (Hips on Heels) - hip to heel line
    if step_state(1) != "not_visible":
        draw_side_line("hip", "heel", color_for(1), thick=3)

    # Step 2 (Torso Folded) - shoulder->hip->knee
    if step_state(2) != "not_visible":
        draw_side_line("shoulder", "hip", color_for(2), thick=5)
        draw_side_line("hip", "knee", color_for(2), thick=5)

    # Step 3 (Arms Extended) - shoulder->elbow->wrist
    if step_state(3) != "not_visible":
        draw_side_line("shoulder", "elbow", color_for(3))
        draw_side_line("elbow", "wrist", color_for(3))

    # Step 4 (Spine Lengthened) - hip->shoulder->wrist
    if step_state(4) != "not_visible":
        draw_side_line("hip", "shoulder", color_for(4), thick=4)
        draw_side_line("shoulder", "wrist", color_for(4), thick=4)

    # Step 5 (Forehead Down) - nose dot
    if step_state(5) != "not_visible" and is_vis("nose"):
        dot(get_pt("nose"), color_for(5), r=10)

    # Step 6 (Shoulders Relaxed) - ear-to-shoulder
    if step_state(6) != "not_visible":
        draw_side_line("ear", "shoulder", color_for(6), thick=3)

    # White dots on all visible landmarks
    for name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle",
                 "left_heel", "right_heel"]:
        if is_vis(name):
            dot(get_pt(name), (255, 255, 255), r=4)

    annotated_path = os.path.join(save_dir, "annotated_full.jpg")
    cv2.imwrite(annotated_path, annotated)
    paths["annotated"] = annotated_path

    # Helper to get visible points for crop bounds
    def visible_pts(names):
        pts = []
        for name in names:
            if is_vis(name):
                pts.append(get_pt(name))
        return pts

    # Step 1: feet area
    pts = visible_pts([f"{side}_hip", f"{side}_knee",
                       f"{side}_ankle", f"{side}_heel"])
    if not pts:
        other = "right" if side == "left" else "left"
        pts = visible_pts([f"{other}_hip", f"{other}_knee",
                           f"{other}_ankle", f"{other}_heel"])
    crop = _crop_with_padding(annotated, pts, 0.12, 0.12) if pts else annotated.copy()
    p1 = os.path.join(save_dir, "step1_hips_on_heels.jpg")
    cv2.imwrite(p1, crop); paths["step_1"] = p1

    # Step 2: torso fold
    pts = visible_pts([f"{side}_shoulder", f"{side}_hip", f"{side}_knee"])
    if not pts:
        other = "right" if side == "left" else "left"
        pts = visible_pts([f"{other}_shoulder", f"{other}_hip", f"{other}_knee"])
    crop = _crop_with_padding(annotated, pts, 0.10, 0.10) if pts else annotated.copy()
    p2 = os.path.join(save_dir, "step2_torso_fold.jpg")
    cv2.imwrite(p2, crop); paths["step_2"] = p2

    # Step 3: arms
    pts = visible_pts([f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"])
    if not pts:
        other = "right" if side == "left" else "left"
        pts = visible_pts([f"{other}_shoulder", f"{other}_elbow", f"{other}_wrist"])
    crop = _crop_with_padding(annotated, pts, 0.10, 0.15) if pts else annotated.copy()
    p3 = os.path.join(save_dir, "step3_arms_extended.jpg")
    cv2.imwrite(p3, crop); paths["step_3"] = p3

    # Step 4: spine line
    pts = visible_pts([f"{side}_hip", f"{side}_shoulder", f"{side}_wrist"])
    if not pts:
        other = "right" if side == "left" else "left"
        pts = visible_pts([f"{other}_hip", f"{other}_shoulder", f"{other}_wrist"])
    crop = _crop_with_padding(annotated, pts, 0.10, 0.10) if pts else annotated.copy()
    p4 = os.path.join(save_dir, "step4_spine_lengthened.jpg")
    cv2.imwrite(p4, crop); paths["step_4"] = p4

    # Step 5: head and wrists
    pts = visible_pts(["nose"]) + visible_pts([f"{side}_wrist"])
    if len(pts) < 2:
        other = "right" if side == "left" else "left"
        pts = visible_pts(["nose"]) + visible_pts([f"{other}_wrist"])
    crop = _crop_with_padding(annotated, pts, 0.15, 0.20) if pts else annotated.copy()
    p5 = os.path.join(save_dir, "step5_forehead_down.jpg")
    cv2.imwrite(p5, crop); paths["step_5"] = p5

    # Step 6: shoulder and ear
    pts = visible_pts([f"{side}_shoulder", f"{side}_ear"])
    if not pts:
        other = "right" if side == "left" else "left"
        pts = visible_pts([f"{other}_shoulder", f"{other}_ear"])
    crop = _crop_with_padding(annotated, pts, 0.15, 0.20) if pts else annotated.copy()
    p6 = os.path.join(save_dir, "step6_shoulders_relaxed.jpg")
    cv2.imwrite(p6, crop); paths["step_6"] = p6

    return paths


# -----------------------------------------------------------------------------
# Result helper for non-Child's-Pose inputs
# -----------------------------------------------------------------------------
def _not_childs_pose_result(best_frame_path=None, source_image_path=None,
                            annotated_path=None, mode="video"):
    return {
        "final_score": 0,
        "issues": ["This does not look like Child's Pose - "
                   "you appear to be standing or in a different pose"],
        "steps": [],
        "best_frame_path": best_frame_path or source_image_path,
        "annotated_path": annotated_path,
        "step_image_paths": {},
        "low_quality_warning": True,
        "low_quality_message": (
            "We did not detect Child's Pose. "
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

            if is_floor_pose(lms, w, h):
                floor_pose_frames += 1
            else:
                standing_pose_frames += 1
                continue

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
