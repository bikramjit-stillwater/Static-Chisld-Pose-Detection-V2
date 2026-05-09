import os
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------------------
# Rule-based fallback (used when no Gemini key, or the API call fails)
# -----------------------------------------------------------------------------
def get_rule_based_feedback(score, issues, steps=None):
    lines = []

    if score >= 85:
        lines.append("Excellent work! Your Tadasana looks stable and well-aligned overall.")
    elif score >= 70:
        lines.append("Good attempt. Your pose is mostly correct, with a few areas to refine.")
    elif score >= 50:
        lines.append("Decent start. Focus on the foundational alignment cues to make your Tadasana stronger.")
    elif score >= 30:
        lines.append("Your pose is off in several places. Re-watch the steps and try again slowly.")
    else:
        lines.append("This does not look like a correct Tadasana yet. Please review the 6 steps and start fresh.")

    lines.append("")

    if steps:
        lines.append("Step-by-step assessment:")
        for s in steps:
            status = "OK" if s["passed_overall"] else "Needs work"
            lines.append(
                f"{s['step']}. {s['name']}: {status}  "
                f"(score {s['average_score']}/100, "
                f"failed in {s['fail_rate_percent']}% of frames)"
            )
            if s["issue"]:
                lines.append(f"   -> {s['issue']}")
                lines.append(f"   Cue: {s['cue']}")
        lines.append("")

    lines.append(
        "Breathe steadily, lengthen through the crown of your head, "
        "and stand like a mountain - rooted, balanced, aware."
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Gemini-powered feedback
# -----------------------------------------------------------------------------
def _build_step_summary(steps):
    if not steps:
        return "No detailed step data available."

    lines = []
    for s in steps:
        status = "PASSED" if s["passed_overall"] else "NEEDS WORK"
        lines.append(
            f"Step {s['step']} - {s['name']}: {status} "
            f"(score {s['average_score']}/100, "
            f"failed in {s['fail_rate_percent']}% of frames)"
        )
        if s["issue"]:
            lines.append(f"   Issue observed: {s['issue']}")
        lines.append(f"   Alignment cue: {s['cue']}")
    return "\n".join(lines)


def get_gemini_feedback(score, issues, steps=None):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return get_rule_based_feedback(score, issues, steps)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        step_summary = _build_step_summary(steps)
        issues_text = (
            "\n".join([f"- {issue}" for issue in issues])
            if issues else "- None significant"
        )

        prompt = f"""
You are an honest, kind, knowledgeable yoga teacher trained in traditional
Hatha and Vinyasa yoga.

A student just performed Tadasana (Mountain Pose). It was analysed with the
6-step ground-truth method (stance, grounding, legs/knees, spine, shoulders/arms,
head/neck). Be HONEST in your feedback - if the score is low, do not pretend
the pose was good.

OVERALL SCORE: {score}/100

SCORING GUIDE (use this to set your tone):
  90-100  Excellent - small refinements only
  75-89   Good - a couple of clear corrections
  55-74   Mixed - several real issues, gentle but firm corrections needed
  30-54   Poor - the pose is not Tadasana yet, walk the student through basics
   0-29   Very poor - explicitly say it does not look like Tadasana yet

STEP-BY-STEP REPORT:
{step_summary}

KEY ISSUES OBSERVED:
{issues_text}

Now write feedback in EXACTLY this format:

Summary:
<one sentence that matches the score band - encouraging if high, honest and
direct if low. Do NOT say "excellent" if the score is below 75.>

Step-by-Step Feedback:
1. Stance: <one short sentence based on the actual data>
2. Grounding: <one short sentence>
3. Legs & Knees: <one short sentence - if knees LOCKED, warn against
   hyperextension; if BENT, encourage softly straightening>
4. Spine: <one short sentence>
5. Shoulders & Arms: <one short sentence - if arms are RAISED, tell the
   student to bring them down beside the body>
6. Head & Neck: <one short sentence>

Top 3 Priorities to Improve:
1. <most important fix>
2. <second priority>
3. <third priority>

Motivation:
<one short, warm closing sentence>

RULES:
- Match the tone to the score band above. NEVER call a sub-50 pose "good".
- Reference the actual scores or fail rates when relevant.
- If a step passed (above 80), give a brief affirmation, not a correction.
- If a step failed badly (below 40), be clear and specific about what to fix.
- Keep each line short and practical.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()

        return get_rule_based_feedback(score, issues, steps)

    except Exception as e:
        return (
            get_rule_based_feedback(score, issues, steps)
            + f"\n\n[Gemini fallback used because API call failed: {str(e)}]"
        )