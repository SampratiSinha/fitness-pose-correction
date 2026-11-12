"""
guidance.py - coaching content for the platform (no heavy imports, safe to import anywhere)

  get_cue()      turns "this joint is too high/low in this phase" into a coaching instruction
  grade()        score -> Excellent / Good / Needs work / Poor
  EXERCISE_GUIDE setup, steps, key points, common mistakes and what the system checks
  SCORE_EXPLANATION  plain-language description of the score function (shown in the dashboard)

Direction: "high" = the user's angle is LARGER than the reference, "low" = SMALLER.
"""

GRADE_BANDS = [(0.85, "Excellent"), (0.70, "Good"), (0.50, "Needs work"), (0.0, "Poor")]


def grade(score):
    if score is None:
        return "-"
    for limit, name in GRADE_BANDS:
        if score >= limit:
            return name
    return "Poor"


# CUES[exercise][joint][direction] -> {phase: text, "any": fallback}
CUES = {
    "squat": {
        "knee": {
            "high": {"down": "Sit deeper: bend your knees until your thighs are close to parallel with the floor.",
                     "up": "Soften your knees slightly at the top instead of locking them out."},
            "low": {"down": "You are lower than the reference. Keep your heels down and your knees tracking over your toes.",
                    "up": "Stand all the way up: straighten your knees at the top of the rep."},
        },
        "hip": {
            "high": {"down": "Push your hips back and down, as if sitting into a chair.",
                     "any": "Keep a slight hip hinge instead of standing too stiff."},
            "low": {"down": "Your hips are folding too much: lift your chest and keep your back long.",
                    "up": "Open your hips fully at the top: squeeze your glutes."},
        },
        "torso_lean": {
            "high": {"any": "Keep your chest up: you are leaning too far forward."},
            "low": {"any": "Let your torso tilt forward a little to stay balanced over mid-foot."},
        },
    },
    "pushup": {
        "elbow": {
            "high": {"down": "Lower your chest further: bend your elbows towards 90 degrees.",
                     "up": "Do not lock your elbows hard at the top."},
            "low": {"down": "Do not drop too far: stop when your chest is just above the floor.",
                    "up": "Press all the way up: straighten your arms at the top."},
        },
        "body_line": {
            "low": {"any": "Keep one straight line from head to heels: tighten your core and glutes (hips are sagging or piking)."},
            "high": {"any": "Keep a neutral spine: do not arch your lower back."},
        },
    },
    "bicep_curl": {
        "elbow": {
            "high": {"curled": "Curl higher: bring the weight up towards your shoulder.",
                     "extended": "Keep a slight bend in the elbow at the bottom to protect the joint."},
            "low": {"curled": "Do not over-curl: stop when the weight is near your shoulder.",
                    "extended": "Lower fully: straighten your arm at the bottom of each rep."},
        },
        "torso_lean": {
            "high": {"any": "Stop swinging: keep your torso still and upright."},
            "low": {"any": "Stand tall: avoid leaning back to lift the weight."},
        },
    },
    "shoulder_press": {
        "elbow": {
            "high": {"down": "Lower the weight until your elbows are at about 90 degrees.",
                     "up": "Keep a soft bend at the top instead of locking out hard."},
            "low": {"up": "Press all the way up: straighten your arms overhead.",
                    "down": "Do not drop too far: keep your elbows at about shoulder height."},
        },
        "torso_lean": {
            "high": {"any": "Keep your ribs down and your body upright: avoid leaning back."},
            "low": {"any": "Stay tall and brace your core as you press."},
        },
    },
    "plank": {
        "body_line": {
            "low": {"any": "Your hips are sagging or piking: squeeze your glutes and abs to form one line from head to heels."},
            "high": {"any": "Keep your body in one straight line without arching your back."},
        },
        "hip": {
            "low": {"any": "Lengthen your body: do not let your hips sink towards the floor."},
            "high": {"any": "Lower your hips slightly so they line up with your shoulders and ankles."},
        },
    },
}

JOINT_LABELS = {"knee": "knee angle", "elbow": "elbow angle", "hip": "hip angle",
                "body_line": "body alignment", "torso_lean": "torso lean"}


def get_cue(exercise, joint, phase, direction, value, reference):
    """Coaching instruction for one off-target joint, with the numbers for transparency."""
    options = CUES.get(exercise, {}).get(joint, {}).get(direction, {})
    text = options.get(phase) or options.get("any")
    if text is None:
        word = "increase" if direction == "low" else "reduce"
        text = f"Adjust your {JOINT_LABELS.get(joint, joint)}: {word} it towards the reference."
    return text


EXERCISE_GUIDE = {
    "squat": {
        "title": "Squat",
        "view": "Side view, full body in frame.",
        "setup": "Feet about shoulder-width apart, toes slightly out, chest up, core braced.",
        "steps": ["Push your hips back and bend your knees to lower.",
                  "Go down until your thighs are close to parallel with the floor.",
                  "Drive through your heels to stand back up and finish tall."],
        "key_points": ["Knees track over the toes", "Chest stays lifted", "Heels stay on the floor"],
        "mistakes": ["Shallow depth", "Leaning too far forward", "Knees caving inwards"],
        "checks": "Knee angle, hip angle and torso lean in the up and down positions. Counts one rep per full down-and-up cycle.",
    },
    "pushup": {
        "title": "Push-up",
        "view": "Side view, full body in frame.",
        "setup": "Hands under the shoulders, body in one straight line from head to heels.",
        "steps": ["Bend your elbows to lower your chest towards the floor.",
                  "Keep your core tight so your hips do not sag.",
                  "Press back up until your arms are straight."],
        "key_points": ["Straight body line", "Elbows near 90 degrees at the bottom", "Full lockout-free extension at the top"],
        "mistakes": ["Sagging hips", "Piked hips", "Half repetitions"],
        "checks": "Elbow angle and body-line angle in the up and down positions. Counts one rep per full cycle.",
    },
    "bicep_curl": {
        "title": "Bicep curl",
        "view": "Side view, full body in frame.",
        "setup": "Stand tall, elbows close to your sides, weights hanging with arms nearly straight.",
        "steps": ["Curl the weight up towards your shoulder without moving your elbows forward.",
                  "Squeeze at the top.",
                  "Lower slowly until your arm is nearly straight."],
        "key_points": ["Elbows stay pinned to your sides", "Torso stays still", "Slow, controlled lowering"],
        "mistakes": ["Swinging the body", "Half curls", "Leaning back"],
        "checks": "Elbow angle in the extended and curled positions, plus torso lean. Counts one rep per full curl.",
    },
    "shoulder_press": {
        "title": "Shoulder press",
        "view": "Side view, full body in frame.",
        "setup": "Stand tall with weights at shoulder height, elbows at about 90 degrees, core braced.",
        "steps": ["Press the weights straight overhead until your arms are extended.",
                  "Keep your ribs down and avoid arching your back.",
                  "Lower under control back to shoulder height."],
        "key_points": ["Wrists above elbows", "Ribs down, no back arch", "Full press overhead"],
        "mistakes": ["Leaning back", "Half press", "Dropping the weights too low"],
        "checks": "Elbow angle in the up and down positions, plus torso lean. Counts one rep per full press.",
    },
    "plank": {
        "title": "Plank",
        "view": "Side view, full body in frame.",
        "setup": "Forearms or hands on the floor, shoulders over elbows, legs extended.",
        "steps": ["Form one straight line from head to heels.",
                  "Brace your abs and squeeze your glutes.",
                  "Hold the position and keep breathing."],
        "key_points": ["Hips level with shoulders and ankles", "Neck neutral", "Core tight"],
        "mistakes": ["Sagging hips", "Hips too high", "Holding your breath"],
        "checks": "Body-line and hip angles. Hold time counts only while your form score stays at 60% or above.",
    },
}

SCORE_EXPLANATION = """
**Per-joint score.** For each key joint the system measures your angle `a` and compares it with the reference mean `m` for the
current phase. The tolerance is `t = max(2 x standard deviation, 10 degrees)`.

`joint score = clip( 1 - max(0, |a - m| - t) / (2t), 0, 1 )`

Inside the tolerance the joint scores 1.0; the score falls linearly to 0 at three times the tolerance.

**Overall score.** `overall = 0.5 x mean(joint scores) + 0.5 x worst joint score`. Blending in the worst joint stops one badly
wrong joint from being hidden by several correct ones.

**Grade.** 85% and above is *Excellent*, 70% to 84% is *Good*, 50% to 69% is *Needs work*, below 50% is *Poor*.

**Feedback.** Joints scoring below 60% generate a coaching cue, most serious first. Each cue says what to change and shows your angle against the reference.

**Phase and reps.** The phase (for example up or down) is the reference phase whose primary angle is closest to yours. A rep is counted when the smoothed
primary angle goes high, then low, then high again, using thresholds derived from the reference photos.
"""
