import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from build_database_images import collect_images, extract_row, primary_angle, EXERCISES
from comparison_engine import PoseScorer

mp_pose = mp.solutions.pose

def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, default=root / "data")
    ap.add_argument("--profiles", type=Path, default=root / "database" / "exercise_profiles.json")
    ap.add_argument("--out", type=Path, default=root / "database" / "evaluation")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2])
    args = ap.parse_args()

    scorer = PoseScorer(args.profiles)
    items = collect_images(args.test_dir)
    if not items:
        raise SystemExit(f"No images found in {args.test_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=args.complexity,
                      min_detection_confidence=0.5) as pose:
        for ex, label, folder_phase, path in items:
            if ex not in scorer.profiles:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            row, _ = extract_row(pose, img, 0.5)
            if row is None:
                print(f"  skipped (no pose): {path.name}")
                continue
            pa = primary_angle(row, EXERCISES[ex])
            if np.isnan(pa):
                print(f"  skipped (joint hidden): {path.name}")
                continue
            res = scorer.score(row, ex, phase=folder_phase or None, primary_angle=pa)
            score = res.get("angle_score")
            if score is None:
                continue
            rows.append({"image": str(path.relative_to(args.test_dir)).replace("\\", "/"),
                         "exercise": ex, "true_label": label, "phase": res["phase"],
                         "score": score, "predicted": "correct" if score >= args.threshold else "incorrect",
                         "feedback": " | ".join(res["feedback"])})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nothing could be scored.")
    df["hit"] = df["true_label"] == df["predicted"]
    df.to_csv(args.out / "scores.csv", index=False)

    by_label = df.groupby("true_label")["score"].agg(["mean", "std", "count"]).round(3)
    per_ex = df.groupby("exercise").agg(
        n=("hit", "size"), accuracy=("hit", "mean"),
        mean_correct=("score", lambda s: s[df.loc[s.index, "true_label"] == "correct"].mean()),
        mean_incorrect=("score", lambda s: s[df.loc[s.index, "true_label"] == "incorrect"].mean())).round(3)
    tp = int(((df.true_label == "incorrect") & (df.predicted == "incorrect")).sum())
    fn = int(((df.true_label == "incorrect") & (df.predicted == "correct")).sum())
    fp = int(((df.true_label == "correct") & (df.predicted == "incorrect")).sum())
    tn = int(((df.true_label == "correct") & (df.predicted == "correct")).sum())

    summary = {
        "test_dir": str(args.test_dir), "threshold": args.threshold, "n_images": int(len(df)),
        "overall_accuracy": round(float(df["hit"].mean()), 3),
        "mean_score_correct": float(by_label.loc["correct", "mean"]) if "correct" in by_label.index else None,
        "mean_score_incorrect": float(by_label.loc["incorrect", "mean"]) if "incorrect" in by_label.index else None,
        "confusion": {"incorrect_caught": tp, "incorrect_missed": fn,
                      "correct_flagged_wrongly": fp, "correct_passed": tn},
        "per_exercise": per_ex.to_dict(orient="index"),
        "note": "In-sample if test_dir is the database folder; use held-out photos for the main result.",
    }
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Mean form score by true label ===")
    print(by_label.to_string())
    print(f"\n=== Accuracy at threshold {args.threshold} ===  overall: {summary['overall_accuracy']*100:.1f}%")
    print(f"incorrect caught {tp} | incorrect missed {fn} | correct wrongly flagged {fp} | correct passed {tn}")
    print("\n=== Per exercise ===")
    print(per_ex.to_string())
    print(f"\nSaved: {args.out/'scores.csv'}\nSaved: {args.out/'summary.json'}")

if __name__ == "__main__":
    main()