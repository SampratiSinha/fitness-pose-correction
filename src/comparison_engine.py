"""
comparison_engine.py - scoring, coaching feedback and rep counting

PoseScorer.score() returns: phase, angle_score (0-1), grade, per_joint breakdown, a ranked
"guidance" list (most serious first) and plain-text "feedback" lines.
RepCounter counts repetitions from the smoothed primary angle (used by live_session and video_analysis).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from guidance import get_cue, grade

JOINT_LABELS = {
    "knee": "knee bend", "elbow": "elbow bend", "hip": "hip fold",
    "body_line": "body alignment (hips sagging/piking)", "torso_lean": "torso lean",
}

class PoseScorer:
    def __init__(self, profiles_path):
        with open(profiles_path) as f:
            self.profiles = json.load(f)

    def guess_phase(self, exercise, primary_angle):
        prof = self.profiles[exercise]
        if prof["static"]:
            return "hold"
        best_phase, best_dist = None, float("inf")
        for phase, stats in prof["phases"].items():
            j = prof["primary_joint"]
            if j not in stats:
                continue
            d = abs(primary_angle - stats[j]["mean"])
            if d < best_dist:
                best_dist, best_phase = d, phase
        return best_phase

    def score(self, angles, exercise, phase=None, primary_angle=None):
        if exercise not in self.profiles:
            raise ValueError(f"Unknown exercise '{exercise}'. Known: {list(self.profiles)}")
        prof = self.profiles[exercise]
        if phase is None:
            phase = self.guess_phase(exercise, primary_angle if primary_angle is not None
                                     else angles.get(prof["primary_joint"], 0))
        stats = prof["phases"].get(phase, {})
        if not stats:
            return {"phase": phase, "angle_score": None, "feedback": ["No reference data for this phase."]}

        per_joint, issues = {}, []
        for joint, s in stats.items():
            if joint not in angles or np.isnan(angles[joint]):
                continue
            mean, tol = s["mean"], s.get("suggested_tolerance_deg", max(2 * s["std"], 10))
            dev = abs(angles[joint] - mean)
            sc = float(np.clip(1 - max(0, dev - tol) / (2 * tol), 0, 1))
            per_joint[joint] = {"value": round(angles[joint], 1), "reference_mean": round(mean, 1),
                               "deviation_deg": round(dev, 1), "score": round(sc, 2)}
            if sc < 0.6:
                issues.append((sc, joint, "high" if angles[joint] > mean else "low", float(angles[joint]), float(mean)))

        if per_joint:
            sc = [v["score"] for v in per_joint.values()]
            overall = 0.5 * float(np.mean(sc)) + 0.5 * float(np.min(sc))
        else:
            overall = None
        guidance = []
        for sc, joint, direction, value, mean in sorted(issues):
            cue = get_cue(exercise, joint, phase, direction, value, mean)
            guidance.append({"joint": joint, "score": round(sc, 2), "direction": direction, "cue": cue,
                             "value": round(value, 1), "reference": round(mean, 1),
                             "text": f"{cue} ({value:.0f} deg vs ~{mean:.0f} deg)"})
        score_value = round(overall, 2) if overall is not None else None
        return {"phase": phase, "angle_score": score_value, "grade": grade(score_value),
                "per_joint": per_joint, "guidance": guidance,
                "feedback": [g["text"] for g in guidance] or ["Good form - within expected range."]}

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        a, b = np.asarray(vec_a, dtype=float), np.asarray(vec_b, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1, 1))


class RepCounter:
    """Counts a rep each time the primary angle goes high -> low -> high (with smoothing)."""

    def __init__(self, low_thr, high_thr, alpha=0.4):
        self.low, self.high, self.alpha = low_thr, high_thr, alpha
        self.reset()

    def reset(self):
        self.reps, self.state, self.smooth = 0, None, None

    def update(self, angle):
        self.smooth = angle if self.smooth is None else self.alpha * angle + (1 - self.alpha) * self.smooth
        a = self.smooth
        if a >= self.high:
            if self.state == "low":
                self.reps += 1
            self.state = "high"
        elif a <= self.low and self.state == "high":
            self.state = "low"
        return self.reps


def _standalone():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cv2
    import mediapipe as mp
    from build_database_images import compute_angles, primary_angle, EXERCISES, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--exercise", required=True)
    ap.add_argument("--profiles", default="../database/exercise_profiles.json")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        raise SystemExit("No person detected.")
    lms = res.pose_landmarks.landmark
    lm_px = np.array([[l.x * w, l.y * h] for l in lms])
    vis = np.array([l.visibility for l in lms])
    angles = compute_angles(lm_px, vis)

    scorer = PoseScorer(args.profiles)
    cfg = EXERCISES[args.exercise]
    pa = primary_angle(angles, cfg)
    result = scorer.score(angles, args.exercise, primary_angle=pa)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    _standalone()