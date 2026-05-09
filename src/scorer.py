"""
Tadasana (Mountain Pose) validator - Arms Overhead Version.

Target pose: standing tall with arms raised straight overhead, palms together,
feet together or hip-width, body stretched upward like a mountain.

6 ground-truth steps, each scored 0-100 with quadratic curves.
Compound penalties applied when multiple steps fail.
"""

import math


# -----------------------------------------------------------------------------
# Geometry helper
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


# =============================================================================
# Step 1 - Stance: feet together or hip-width (NOT wider)
# =============================================================================
def check_stance(features):
    ratio = features["stance_ratio"]
    if ratio <= 1.1:
        return {
            "step": 1, "name": "Stance",
            "passed": True, "score": 100.0, "issue": None,
            "cue": "Stand with feet together or at hip-distance",
        }
    score = score_value(ratio - 1.1, 0.0, 0.6, "quadratic")
    return {
        "step": 1, "name": "Stance",
        "passed": ratio <= 1.25, "score": score,
        "issue": "Feet are too far apart - bring them to hip-width or together",
        "cue": "Stand with feet together or at hip-distance",
    }


# =============================================================================
# Step 2 - Body Balance: weight centered, no leaning
# =============================================================================
def check_body_balance(features):
    body_lean = features["body_lean"]
    score = score_value(body_lean, 0.025, 0.09, "quadratic")
    passed = body_lean <= 0.04
    return {
        "step": 2, "name": "Body Balance",
        "passed": passed, "score": score,
        "issue": None if passed else "Body is leaning - distribute weight evenly across both feet",
        "cue": "Press the four corners of each foot into the floor evenly",
    }


# =============================================================================
# Step 3 - Legs & Knees: soft, NOT locked, NOT bent
# =============================================================================
def check_legs_knees(features):
    left = features["left_knee_bend"]
    right = features["right_knee_bend"]
    bent = left < 168 or right < 168
    locked = left > 178 or right > 178

    if not bent and not locked:
        return {
            "step": 3, "name": "Legs & Knees",
            "passed": True, "score": 100.0, "issue": None,
            "cue": "Lift kneecaps gently - straight but never locked",
        }
    if locked and not bent:
        worst = max(left, right)
        return {
            "step": 3, "name": "Legs & Knees",
            "passed": False,
            "score": score_value(worst - 178, 0.0, 6.0, "quadratic"),
            "issue": "Knees are locked - keep them soft and active, not rigid",
            "cue": "Lift kneecaps gently - straight but never locked",
        }
    if bent and not locked:
        worst = min(left, right)
        return {
            "step": 3, "name": "Legs & Knees",
            "passed": False,
            "score": score_value(168 - worst, 0.0, 18.0, "quadratic"),
            "issue": "Knees are bent - gently straighten without locking",
            "cue": "Lift kneecaps gently - straight but never locked",
        }
    return {
        "step": 3, "name": "Legs & Knees",
        "passed": False, "score": 30.0,
        "issue": "One knee bent and the other locked - aim for soft and even",
        "cue": "Lift kneecaps gently - straight but never locked",
    }


# =============================================================================
# Step 4 - Spine: vertical and lengthened
# =============================================================================
def check_spine(features):
    spine_tilt = features["spine_tilt"]
    score = score_value(spine_tilt, 2.5, 11.0, "quadratic")
    passed = spine_tilt <= 5.0
    return {
        "step": 4, "name": "Spine",
        "passed": passed, "score": score,
        "issue": None if passed else "Spine is not vertical - tailbone down, crown of head up",
        "cue": "Lengthen the spine - tailbone tucks down, crown lifts up",
    }


# =============================================================================
# Step 5 - Shoulders & Arms: ARMS RAISED OVERHEAD (the cartoon pose)
#  Sub-checks:
#    (a) Wrists ABOVE shoulders (arms raised, not hanging)
#    (b) Elbows straight (arms not bent)
#    (c) Arms close together overhead (palms touching/near)
#    (d) Symmetric (both arms equally raised)
# =============================================================================
def check_shoulders_arms(features):
    # (a) Arms raised - wrist drop should be NEGATIVE (above shoulders)
    # arm_drop: 0 = at shoulder, negative = above, positive = below
    v_left = features["left_arm_drop"]
    v_right = features["right_arm_drop"]
    worst_v = max(v_left, v_right)  # the lower-hanging arm

    # Ideal: drop <= -0.25 (wrist clearly above shoulder, near head/above)
    # Bad: drop > 0 (wrist below shoulder = arms hanging)
    if worst_v <= -0.25:
        raised_score = 100.0
    elif worst_v <= 0.0:
        # Between shoulder and ideal: partial credit
        raised_score = score_value(0.0 - worst_v, 0.25, 0.0, "quadratic")
        # Note: this gives 0 at worst_v=0.0, 100 at worst_v=-0.25
        # Need different formula - reverse it:
        # The closer to -0.25 (or below), the better
        raised_score = round(100.0 * ((-worst_v) / 0.25) ** 2, 1)
    else:
        # Wrist below shoulder - arms hanging - NOT Tadasana
        raised_score = 0.0

    # (b) Elbows straight
    e_left = features["left_elbow_angle"]
    e_right = features["right_elbow_angle"]
    worst_elbow = min(e_left, e_right)
    # Ideal: 165-180 (mostly straight). Bad: < 130
    if worst_elbow >= 165:
        elbow_score = 100.0
    else:
        elbow_score = score_value(165 - worst_elbow, 0.0, 35.0, "quadratic")

    # (c) Arms close together (horizontal distance between wrists, normalized)
    arm_closeness = features["arm_closeness"]
    # Ideal: <= 0.05 (wrists touching/very close)
    # Acceptable: <= 0.20 (parallel arms shoulder-width)
    closeness_score = score_value(arm_closeness, 0.05, 0.40, "quadratic")

    # (d) Symmetry (left vs right arm height difference)
    asymmetry = abs(v_left - v_right)
    symmetry_score = score_value(asymmetry, 0.04, 0.20, "quadratic")

    # Weighted combine (raised position is most important)
    score = round(
        raised_score * 0.45 +
        elbow_score * 0.25 +
        closeness_score * 0.15 +
        symmetry_score * 0.15,
        1,
    )

    issues = []
    if worst_v > -0.05:
        issues.append("Arms are not raised - stretch them straight up overhead")
    elif worst_v > -0.20:
        issues.append("Reach arms higher - extend fully overhead")
    if worst_elbow < 160:
        issues.append("Elbows are bent - straighten the arms")
    if arm_closeness > 0.30:
        issues.append("Bring the arms closer together overhead")
    if asymmetry > 0.10:
        issues.append("One arm is higher than the other - keep them even")

    passed = len(issues) == 0
    issue = " - ".join(issues) if issues else None

    return {
        "step": 5, "name": "Shoulders & Arms",
        "passed": passed, "score": score,
        "issue": issue,
        "cue": "Stretch arms straight up overhead, palms together, elbows straight",
    }


# =============================================================================
# Step 6 - Head & Neck: balanced between the raised arms
# =============================================================================
def check_head_neutral(features):
    head_offset = features["head_offset"]
    score = score_value(head_offset, 0.03, 0.11, "quadratic")
    passed = head_offset <= 0.06
    return {
        "step": 6, "name": "Head & Neck",
        "passed": passed, "score": score,
        "issue": None if passed else "Head is tilting - keep it balanced, gaze forward",
        "cue": "Keep the head balanced between the arms, gaze soft and forward",
    }


# Step weights (sum to 1.0)
STEP_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.25, 6: 0.10}


def validate_tadasana(features):
    """Run 6-step validation with compound penalties."""
    step_results = [
        check_stance(features),
        check_body_balance(features),
        check_legs_knees(features),
        check_spine(features),
        check_shoulders_arms(features),
        check_head_neutral(features),
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


def score_tadasana(features):
    report = validate_tadasana(features)
    return report["final_score"], report["issues"]