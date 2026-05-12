"""
Child's Pose validator.

Source: https://rishikeshyogavalley.com/utthita-balasana-extended-childs-pose/

Target pose: kneeling forward fold with arms extended out in front, palms down,
forehead resting on the mat, hips on heels, spine lengthened.

6 ground-truth scoring steps, each 0-100 with quadratic curves.

Visibility rule:
  If a body part required for a step is NOT VISIBLE, that step scores 0
  with a "not visible" message. The 0 is included in the weighted average.
"""

import math


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def score_value(deviation, ideal_max, fail_min, curve="quadratic"):
    """Convert deviation into 0-100 score."""
    if deviation <= ideal_max:
        return 100.0
    if deviation >= fail_min:
        return 0.0
    span = fail_min - ideal_max
    over = deviation - ideal_max
    progress = over / span
    if curve == "quadratic":
        return round(100.0 * (1.0 - progress) ** 2, 1)
    return round(100.0 * (1.0 - progress), 1)


# -----------------------------------------------------------------------------
# "Not visible" helper
# -----------------------------------------------------------------------------
def _not_visible(step_num, name, body_part, cue):
    """Return a step result for a body part that is NOT visible.
    Score is 0 and the issue clearly says it's not visible."""
    return {
        "step": step_num,
        "name": name,
        "passed": False,
        "score": 0.0,
        "issue": f"Cannot evaluate - {body_part} not visible in the frame",
        "cue": cue,
        "not_visible": True,
    }


# =============================================================================
# Step 1 - Hips on Heels
#   Source: "Slowly begin to lower your hips back toward your heels"
#   Measure: hip-to-heel vertical distance, normalized by thigh length.
#   Good: hips sitting close to heels (small distance).
#   Required: hips, knees, ankles
# =============================================================================
def check_hips_on_heels(features, visible=True):
    if not visible:
        return _not_visible(
            1, "Hips on Heels", "hips/legs",
            "Lower your hips back toward your heels",
        )

    # hip_to_heel_ratio: distance from hip to heel / thigh length.
    # In a good Child's Pose, hips are nearly on heels -> ratio close to 0.
    # Sitting up tall (incomplete fold) -> ratio much larger.
    ratio = features["hip_to_heel_ratio"]
    score = score_value(ratio, 0.25, 1.20, "quadratic")
    passed = ratio <= 0.45

    return {
        "step": 1, "name": "Hips on Heels",
        "passed": passed, "score": score,
        "issue": None if passed else
                 "Hips are not sinking back - sit your hips down toward your heels",
        "cue": "Lower your hips back toward your heels",
        "not_visible": False,
    }


# =============================================================================
# Step 2 - Torso Folded Forward
#   Source: "Let your chest and forehead rest on the mat"
#   Measure: angle between torso (hips->shoulders) and thighs (hips->knees).
#   Good: torso folded down close to thighs -> SMALL angle (< 30 deg).
#   Bad: sitting upright -> LARGE angle (~90 deg).
#   Required: shoulders, hips, knees
# =============================================================================
def check_torso_fold(features, visible=True):
    if not visible:
        return _not_visible(
            2, "Torso Folded Forward", "torso (shoulders, hips, knees)",
            "Fold your chest down toward the floor",
        )

    # torso_thigh_angle: angle at hip between torso and thigh.
    # Ideal Child's Pose: torso lying on thighs -> angle ~ 10-25 degrees.
    # Bad (sitting up): angle ~ 80-100 degrees.
    angle = features["torso_thigh_angle"]
    # Smaller is better, so we score the deviation from "perfect" (10 deg)
    score = score_value(angle - 10.0, 15.0, 80.0, "quadratic")
    passed = angle <= 35.0

    return {
        "step": 2, "name": "Torso Folded Forward",
        "passed": passed, "score": score,
        "issue": None if passed else
                 "Torso is not folded down - bring your chest closer to your thighs",
        "cue": "Fold your chest down toward the floor, forehead toward the mat",
        "not_visible": False,
    }


# =============================================================================
# Step 3 - Arms Extended Forward
#   Source: "Stretch your arms out in front of you... keeping them parallel...
#            arms should be actively engaged, reaching forward, not letting elbows bend"
#   Measure:
#     (a) Elbows straight (elbow angles close to 180)
#     (b) Both arms reaching forward
#     (c) Arms parallel (symmetric)
#   Required: shoulders, elbows, wrists
# =============================================================================
def check_arms_extended(features, visible=True):
    if not visible:
        return _not_visible(
            3, "Arms Extended Forward", "arms (shoulders, elbows, wrists)",
            "Stretch your arms out in front, palms down",
        )

    # (a) Elbows straight
    e_left = features["left_elbow_angle"]
    e_right = features["right_elbow_angle"]
    worst_elbow = min(e_left, e_right)
    if worst_elbow >= 165:
        elbow_score = 100.0
    else:
        elbow_score = score_value(165 - worst_elbow, 0.0, 50.0, "quadratic")

    # (b) Arms reaching forward (wrist further from shoulder than elbow is)
    # arm_extension_ratio: distance(shoulder->wrist) / distance(shoulder->elbow)
    # For a fully extended arm this is ~2.0. For a bent arm it's much less.
    ext_left = features["left_arm_extension_ratio"]
    ext_right = features["right_arm_extension_ratio"]
    worst_ext = min(ext_left, ext_right)
    if worst_ext >= 1.85:
        extension_score = 100.0
    else:
        extension_score = score_value(1.85 - worst_ext, 0.0, 0.80, "quadratic")

    # (c) Arms parallel / symmetric (left and right elbow angles similar)
    asymmetry = abs(e_left - e_right)
    symmetry_score = score_value(asymmetry, 5.0, 35.0, "quadratic")

    score = round(
        elbow_score * 0.50 +
        extension_score * 0.30 +
        symmetry_score * 0.20,
        1,
    )

    issues = []
    if worst_elbow < 160:
        issues.append("Elbows are bent - keep your arms actively engaged and straight")
    if worst_ext < 1.70:
        issues.append("Reach your arms further forward")
    if asymmetry > 15:
        issues.append("Arms are not parallel - keep them even")

    passed = len(issues) == 0
    issue = " - ".join(issues) if issues else None

    return {
        "step": 3, "name": "Arms Extended Forward",
        "passed": passed, "score": score,
        "issue": issue,
        "cue": "Stretch your arms out in front, palms down, elbows straight",
        "not_visible": False,
    }


# =============================================================================
# Step 4 - Spine Lengthened
#   Source: "Lengthen through your spine... elongation from fingertips to hips"
#   Measure: how straight a line goes from hips -> shoulders -> wrists.
#   Good: straight elongated line (the body forms a long stretch).
#   Bad: spine curled, hunched.
#   Required: hips, shoulders, wrists
# =============================================================================
def check_spine_lengthened(features, visible=True):
    if not visible:
        return _not_visible(
            4, "Spine Lengthened", "spine line (hips, shoulders, wrists)",
            "Lengthen the spine - reach fingertips forward, sit hips back",
        )

    # spine_line_deviation: angle at shoulders between (hips->shoulders) and
    # (shoulders->wrists). For a perfectly elongated spine, this angle is
    # close to 180 (straight line). Deviation = 180 - angle.
    deviation = features["spine_line_deviation"]
    score = score_value(deviation, 15.0, 70.0, "quadratic")
    passed = deviation <= 30.0

    return {
        "step": 4, "name": "Spine Lengthened",
        "passed": passed, "score": score,
        "issue": None if passed else
                 "Spine is not elongated - reach fingertips forward and sit hips back",
        "cue": "Lengthen the spine - elongate from fingertips to hips",
        "not_visible": False,
    }


# =============================================================================
# Step 5 - Forehead Down (Head Resting)
#   Source: "Forehead on the mat... relax the forehead onto the mat"
#   Measure: head should be close to mat level. We use head height relative
#   to wrist height (since wrists are on the mat).
#   Good: head close to or below wrist level (forehead on mat).
#   Bad: head lifted up.
#   Required: nose, wrists
# =============================================================================
def check_forehead_down(features, visible=True):
    if not visible:
        return _not_visible(
            5, "Forehead Down", "head and arms",
            "Rest your forehead on the mat, or on a block if needed",
        )

    # head_to_mat_ratio:
    #   value = (wrist_y - nose_y) / body_scale
    #   Positive small value = head near mat level (good)
    #   Large negative = head is high up (bad)
    # We allow a small lift (head can be slightly above wrists if neck is long)
    head_lift = features["head_lift_above_mat"]
    # head_lift > 0 means nose is ABOVE wrist level (lifted off mat)
    # head_lift <= 0 means nose is at or below wrist level (good)
    score = score_value(head_lift, 0.05, 0.40, "quadratic")
    passed = head_lift <= 0.12

    return {
        "step": 5, "name": "Forehead Down",
        "passed": passed, "score": score,
        "issue": None if passed else
                 "Head is lifted - rest your forehead on the mat (or a block)",
        "cue": "Relax the forehead onto the mat - use a block if needed",
        "not_visible": False,
    }


# =============================================================================
# Step 6 - Shoulders Relaxed
#   Source: "Shoulders are relaxed and away from the ears"
#   Measure: vertical distance between shoulders and ears, normalized.
#   Good: shoulders DOWN, away from ears (large distance).
#   Bad: shoulders hunched up to ears (small distance, or shoulders above ears).
#   Required: shoulders, ears
# =============================================================================
def check_shoulders_relaxed(features, visible=True):
    if not visible:
        return _not_visible(
            6, "Shoulders Relaxed", "shoulders and ears",
            "Relax your shoulders away from your ears",
        )

    # shoulder_ear_drop: positive = shoulders below ears (relaxed, good)
    #                    near zero = shoulders level with ears (hunched)
    #                    negative = shoulders above ears (very tense)
    drop = features["shoulder_ear_drop"]
    # We want drop to be a reasonable positive value.
    # Ideal: drop >= 0.10 (shoulders well below ears)
    if drop >= 0.10:
        score = 100.0
    elif drop >= 0:
        score = round(100.0 * (drop / 0.10) ** 2, 1)
    else:
        score = 0.0

    passed = drop >= 0.05

    return {
        "step": 6, "name": "Shoulders Relaxed",
        "passed": passed, "score": score,
        "issue": None if passed else
                 "Shoulders are hunched up - relax them down, away from your ears",
        "cue": "Relax your shoulders away from your ears, no tension in the neck",
        "not_visible": False,
    }


# -----------------------------------------------------------------------------
# Step weights - sum to 1.0
# -----------------------------------------------------------------------------
STEP_WEIGHTS = {
    1: 0.20,  # Hips on Heels - foundation
    2: 0.25,  # Torso Folded Forward - the main shape
    3: 0.20,  # Arms Extended Forward - the "extended" part
    4: 0.15,  # Spine Lengthened
    5: 0.10,  # Forehead Down
    6: 0.10,  # Shoulders Relaxed
}


def validate_pose(features, step_visibility=None):
    """Run 6-step validation for Child's Pose with compound penalties.

    Args:
        features: dict from build_features()
        step_visibility: dict {step_num: True/False} from get_step_visibility()
    """
    if step_visibility is None:
        step_visibility = {i: True for i in range(1, 7)}

    step_results = [
        check_hips_on_heels(features, step_visibility.get(1, True)),
        check_torso_fold(features, step_visibility.get(2, True)),
        check_arms_extended(features, step_visibility.get(3, True)),
        check_spine_lengthened(features, step_visibility.get(4, True)),
        check_forehead_down(features, step_visibility.get(5, True)),
        check_shoulders_relaxed(features, step_visibility.get(6, True)),
    ]

    base_score = 0.0
    for s in step_results:
        s["weight"] = STEP_WEIGHTS[s["step"]]
        base_score += s["score"] * s["weight"]

    worst = min(s["score"] for s in step_results)
    very_bad = sum(1 for s in step_results if s["score"] < 20)
    critical = sum(1 for s in step_results if s["score"] < 40)

    final_score = base_score
    if very_bad >= 2:
        final_score *= 0.55
    elif very_bad >= 1:
        final_score *= 0.75
    elif critical >= 2:
        final_score *= 0.85

    if worst < 50:
        final_score = min(final_score, 78.0)
    if worst < 30:
        final_score = min(final_score, 60.0)
    if worst < 15:
        final_score = min(final_score, 45.0)

    final_score = int(round(final_score))
    final_score = max(0, min(100, final_score))

    issues = [s["issue"] for s in step_results if s["issue"]]

    return {
        "final_score": final_score,
        "steps": step_results,
        "issues": issues,
    }


# Backwards-compatible wrappers (old function names)
def validate_tadasana(features, step_visibility=None):
    return validate_pose(features, step_visibility)


def score_tadasana(features, step_visibility=None):
    report = validate_pose(features, step_visibility)
    return report["final_score"], report["issues"]
