"""
video_analysis.py - analyse a recorded exercise video (User view: video, not just a still image)

    python video_analysis.py --video ../my_squats.mp4 --exercise squat

For every sampled frame it runs MediaPipe Pose, scores the form against the reference
profile, and counts reps (or hold time for plank). It reports:
  - reps / hold time, mean score and grade, share of frames with good form (score >= 70%)
  - the score over time, the most frequent coaching cues
  - the best and worst moments, as annotated frames

Outputs (command line): database/video_analysis/<video>_series.csv, _best.jpg, _worst.jpg
The dashboard calls analyze_video() directly.
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from build_database_images import compute_angles, primary_angle, EXERCISES
from comparison_engine import PoseScorer, RepCounter
from guidance import grade

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def analyze_video(video_path, exercise, scorer, sample_every=2, complexity=1, progress=None):
    prof, cfg = scorer.profiles[exercise], EXERCISES[exercise]
    counter = None if prof["static"] else RepCounter(prof["rep_thresholds"]["low"], prof["rep_thresholds"]["high"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open the video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows, cues = [], Counter()
    best, worst = (-1.0, None, 0.0), (2.0, None, 0.0)       # (score, annotated RGB frame, time)
    idx = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=complexity,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # every frame keeps tracking stable
            if idx % sample_every == 0 and res.pose_landmarks:
                h, w = frame.shape[:2]
                lms = res.pose_landmarks.landmark
                angles = compute_angles(np.array([[l.x * w, l.y * h] for l in lms]),
                                        np.array([l.visibility for l in lms]))
                pa = primary_angle(angles, cfg)
                if not np.isnan(pa):
                    result = scorer.score(angles, exercise, primary_angle=pa)
                    sc = result.get("angle_score")
                    if sc is not None:
                        if counter:
                            counter.update(pa)
                        for g in result["guidance"]:
                            cues[g["cue"]] += 1
                        t = idx / fps
                        rows.append({"time_s": round(t, 2), "score": sc, "phase": result["phase"],
                                     "primary_angle": round(float(pa), 1), "reps": counter.reps if counter else None})
                        if sc > best[0] or sc < worst[0]:
                            shot = frame.copy()
                            mp_draw.draw_landmarks(shot, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            shot = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
                            if sc > best[0]:
                                best = (sc, shot, t)
                            if sc < worst[0]:
                                worst = (sc, shot, t)
            idx += 1
            if progress and total:
                progress(min(idx / total, 1.0))
    cap.release()

    if not rows:
        raise ValueError("No person was detected in the video. Use a clear, full-body, side-view recording.")
    df = pd.DataFrame(rows)
    mean_score = float(df["score"].mean())
    n = len(df)
    return {
        "frames_analyzed": n, "duration_s": round(idx / fps, 1), "exercise": exercise,
        "reps": counter.reps if counter else None,
        "hold_s": round(float((df["score"] >= 0.6).sum()) * sample_every / fps, 1) if prof["static"] else None,
        "mean_score": round(mean_score, 3), "grade": grade(mean_score),
        "good_fraction": round(float((df["score"] >= 0.7).mean()), 3),
        "series": df,
        "top_cues": [(c, k, round(k / n, 3)) for c, k in cues.most_common(4)],
        "best": {"score": best[0], "time_s": round(best[2], 1), "image": best[1]},
        "worst": {"score": worst[0], "time_s": round(worst[2], 1), "image": worst[1]},
    }


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--exercise", required=True, choices=list(EXERCISES))
    ap.add_argument("--profiles", type=Path, default=root / "database" / "exercise_profiles.json")
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    scorer = PoseScorer(args.profiles)
    if args.exercise not in scorer.profiles:
        raise SystemExit(f"'{args.exercise}' is not in {args.profiles}. Run: python main.py build")
    r = analyze_video(args.video, args.exercise, scorer, args.sample_every, args.complexity,
                      progress=lambda p: print(f"\r  analysing... {p*100:3.0f}%", end="", flush=True))
    print()
    result = f"reps {r['reps']}" if r["reps"] is not None else f"hold {r['hold_s']}s"
    print(f"{args.exercise}: {result} | mean score {r['mean_score']*100:.0f}% ({r['grade']}) | "
          f"good-form frames {r['good_fraction']*100:.0f}% | {r['frames_analyzed']} frames over {r['duration_s']}s")
    if r["top_cues"]:
        print("Most frequent coaching cues:")
        for cue, k, frac in r["top_cues"]:
            print(f"  {frac*100:3.0f}% of frames: {cue}")
    out = root / "database" / "video_analysis"
    out.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    r["series"].to_csv(out / f"{stem}_series.csv", index=False)
    for key in ("best", "worst"):
        cv2.imwrite(str(out / f"{stem}_{key}.jpg"), cv2.cvtColor(r[key]["image"], cv2.COLOR_RGB2BGR))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
