"""
Feedback generation for Child's Pose.

Uses Google Gemini if GEMINI_API_KEY is set, otherwise falls back to a
deterministic rule-based message.
"""

import os


def _build_step_summary(steps):
    if not steps:
        return "  (no per-step data available)"
    lines = []
    for s in steps:
        if s.get("not_visible"):
            lines.append(
                f"  Step {s['step']} - {s['name']}: NOT VISIBLE in frame "
                f"(cannot evaluate)"
            )
        else:
            score = s.get("average_score", s.get("score", 0))
            status = "PASS" if s.get("passed_overall", s.get("passed", False)) else "FAIL"
            issue = s.get("issue") or "looks good"
            lines.append(
                f"  Step {s['step']} - {s['name']}: {score}/100 [{status}] - {issue}"
            )
    return "\n".join(lines)


def get_rule_based_feedback(score, issues, steps=None):
    """Deterministic fallback feedback."""
    lines = []

    if score is None or score == 0:
        lines.append(
            "Could not evaluate your pose. Please re-record with your full body "
            "visible from the side, in good lighting."
        )
    elif score >= 85:
        lines.append(
            "Excellent! Your Child's Pose shows great relaxation and alignment."
        )
    elif score >= 70:
        lines.append(
            "Good attempt. A few refinements will deepen your Child's Pose."
        )
    elif score >= 50:
        lines.append(
            "Decent start. Focus on sinking hips to heels and lengthening through "
            "your spine and arms."
        )
    elif score >= 30:
        lines.append(
            "Your pose needs work. Review the foundational shape - hips on heels, "
            "torso folded forward, arms extended."
        )
    else:
        lines.append(
            "This does not look like Child's Pose yet. Begin from a "
            "tabletop position and slowly lower your hips back toward your heels."
        )

    if steps:
        lines.append("")
        lines.append("Step-by-step assessment:")
        for s in steps:
            if s.get("not_visible"):
                lines.append(f"{s['step']}. {s['name']}: Not visible - cannot evaluate")
            else:
                status = "OK" if s.get("passed_overall", s.get("passed", False)) else "Needs work"
                lines.append(f"{s['step']}. {s['name']}: {status}")
                if s.get("issue"):
                    lines.append(f"   -> {s['issue']}")

    return "\n".join(lines)


def get_gemini_feedback(score, issues, steps=None):
    """Generate friendly coaching feedback via Gemini. Falls back to rule-based
    if Gemini is unavailable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return get_rule_based_feedback(score, issues, steps)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        step_summary = _build_step_summary(steps)
        issues_text = ("\n".join(f"- {i}" for i in issues)
                       if issues else "- None significant")

        prompt = f"""
You are an honest, kind, knowledgeable yoga teacher.

A student performed Child's Pose.
Be HONEST - if the score is low, do not pretend the pose was good.
If a body part was NOT VISIBLE in the frame, do NOT make up feedback about it.
Simply note that you could not see that part and ask them to re-record.

OVERALL SCORE: {score}/100

SCORING GUIDE:
  90-100  Excellent - small refinements only
  75-89   Good - a couple of clear corrections
  55-74   Mixed - several real issues
  30-54   Poor - the pose is not Child's Pose yet
   0-29   Very poor - explicitly say it does not look like the pose

STEP-BY-STEP REPORT:
{step_summary}

KEY ISSUES OBSERVED:
{issues_text}

The 6 steps of Child's Pose are:
  1. Hips on Heels - hips sink back to rest on heels
  2. Torso Folded Forward - chest folds down toward thighs
  3. Arms Extended Forward - arms reach out in front, elbows straight
  4. Spine Lengthened - long elongation from hips through fingertips
  5. Forehead Down - forehead resting on the mat (or a block)
  6. Shoulders Relaxed - shoulders away from ears, no neck tension

Now write feedback in EXACTLY this format:

Summary: <one honest sentence>

Step-by-Step Feedback:
1. Hips on Heels: <one short line>
2. Torso Folded Forward: <one short line>
3. Arms Extended Forward: <one short line>
4. Spine Lengthened: <one short line>
5. Forehead Down: <one short line>
6. Shoulders Relaxed: <one short line>

Top 3 Priorities to Improve:
1. <priority 1>
2. <priority 2>
3. <priority 3>

Motivation: <one warm closing sentence>

IMPORTANT: For any step marked NOT VISIBLE, your line must say so honestly
(example: "Cannot see your feet - please re-record with your full body in view.")
and you should NOT include that step in the Top 3 Priorities.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()

    except Exception:
        return get_rule_based_feedback(score, issues, steps)
