"""
live_session.py - live Trainer & User session (webcam beside the trainer, scoring, coaching, rep counting)

    python live_session.py --exercise squat
    python live_session.py --exercise squat --trainer-video ../trainer/squat.mp4
    python live_session.py --exercise squat --target-reps 10 --sets 3     (workout mode)
    python live_session.py --exercise squat --record                      (save the session video)

Keys: q = quit, r = reset reps.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from build_database_images import compute_angles, primary_angle, EXERCISES, IMAGE_EXT
from comparison_engine import PoseScorer, RepCounter
from guidance import grade

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
PANEL_H = 480
LOG_FIELDS = ["timestamp", "exercise", "duration_s", "reps_or_hold", "mean_score", "sets_done", "target_reps"]

def append_log(log, row):
    """Append one session row; older logs with fewer columns are upgraded automatically."""
    rows = []
    if log.exists():
        with open(log, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    with open(log, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in LOG_FIELDS})


def fit_height(img, h=PANEL_H):
    return cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h))

def compose(cam, trainer, hud_lines, score):
    left, right = fit_height(cam), fit_height(trainer)
    for panel, label in ((left, "YOU"), (right, "TRAINER")):
        cv2.rectangle(panel, (0, 0), (110, 28), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    body = np.hstack([left, right])
    hud = np.zeros((110, body.shape[1], 3), np.uint8)
    if score is not None:
        colour = (0, 200, 0) if score >= 0.7 else (0, 165, 255) if score >= 0.5 else (0, 0, 255)
        cv2.rectangle(hud, (10, 10), (10 + int(300 * score), 30), colour, -1)
        cv2.rectangle(hud, (10, 10), (310, 30), (200, 200, 200), 1)
    for i, line in enumerate(hud_lines):
        cv2.putText(hud, line, (10, 55 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([body, hud])

def load_trainer(args, data_dir):
    if args.trainer_video:
        cap = cv2.VideoCapture(str(args.trainer_video))
        if cap.isOpened():
            return cap, None
        print("Could not open trainer video, falling back to a reference image.")
    for ex_dir in data_dir.iterdir():
        if ex_dir.is_dir() and ex_dir.name.lower().replace(" ", "").replace("_", "").replace("-", "").startswith(
                args.exercise.replace("_", "")):
            imgs = [p for p in sorted((ex_dir / "correct").rglob("*")) if p.suffix.lower() in IMAGE_EXT] \
                if (ex_dir / "correct").exists() else []
            if imgs:
                return None, cv2.imread(str(imgs[0]))
    blank = np.zeros((PANEL_H, 640, 3), np.uint8)
    cv2.putText(blank, "No trainer media found", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return None, blank

def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", required=True, choices=list(EXERCISES))
    ap.add_argument("--trainer-video", type=Path)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--profiles", type=Path, default=root / "database" / "exercise_profiles.json")
    ap.add_argument("--data", type=Path, default=root / "data")
    ap.add_argument("--record", action="store_true", help="save the session window to database/recordings/")
    ap.add_argument("--target-reps", type=int, default=0, help="reps per set (workout mode)")
    ap.add_argument("--sets", type=int, default=1, help="number of sets (workout mode)")
    args = ap.parse_args()

    scorer = PoseScorer(args.profiles)
    if args.exercise not in scorer.profiles:
        raise SystemExit(f"'{args.exercise}' is not in {args.profiles}. Re-run build_database_images.py.")
    prof, cfg = scorer.profiles[args.exercise], EXERCISES[args.exercise]
    counter = None if prof["static"] else RepCounter(prof["rep_thresholds"]["low"], prof["rep_thresholds"]["high"])

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam (try --camera 1).")
    trainer_cap, trainer_img = load_trainer(args, args.data)

    workout = counter is not None and args.target_reps > 0
    scores, feedback, hold_time, last_t, frame_n = [], ["Get into position"], 0.0, time.time(), 0
    sets_done, total_reps, banner_until, finish_at, writer, rec_path = 0, 0, 0.0, 0.0, None, None
    t_start = time.time()
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cam.isOpened():
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            score, phase = None, "-"
            now = time.time(); dt, last_t = now - last_t, now

            if res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = res.pose_landmarks.landmark
                lm_px = np.array([[l.x * w, l.y * h] for l in lms])
                vis = np.array([l.visibility for l in lms])
                angles = compute_angles(lm_px, vis)
                pa = primary_angle(angles, cfg)
                if not np.isnan(pa):
                    result = scorer.score(angles, args.exercise, primary_angle=pa)
                    score, phase = result.get("angle_score"), result["phase"]
                    if score is not None:
                        scores.append(score)
                        if frame_n % 10 == 0:
                            feedback = [g["cue"] for g in result["guidance"][:2]] or ["Good form - keep going"]
                    if counter and now >= banner_until and not (workout and sets_done >= args.sets):
                        counter.update(pa)           # counting pauses during the 3-second rest banner
                        if workout and counter.reps >= args.target_reps:
                            sets_done += 1
                            total_reps += counter.reps
                            counter.reset()
                            banner_until = now + 3.0
                            if sets_done >= args.sets:
                                finish_at = now + 3.0
                    elif score is not None and score >= 0.6:
                        hold_time += dt
            frame_n += 1

            if trainer_cap is not None:
                ok_t, t_frame = trainer_cap.read()
                if not ok_t:
                    trainer_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_t, t_frame = trainer_cap.read()
                t_panel = t_frame if ok_t else np.zeros((PANEL_H, 640, 3), np.uint8)
            else:
                t_panel = trainer_img.copy()

            if workout:
                counter_txt = f"Set {min(sets_done + 1, args.sets)}/{args.sets}  Reps {counter.reps}/{args.target_reps}"
            elif counter:
                counter_txt = f"Reps: {counter.reps}"
            else:
                counter_txt = f"Hold time: {hold_time:.0f}s"
            score_txt = "--" if score is None else f"{score*100:.0f}% {grade(score)}"
            lines = [f"{args.exercise.upper()}   Phase: {phase}   {counter_txt}   Score: {score_txt}"]
            lines += [f"> {c}"[:115] for c in feedback]
            if time.time() < banner_until:
                lines[0] = (f"SET {sets_done} COMPLETE - take a breath" if sets_done < args.sets
                            else "WORKOUT COMPLETE - great job!")
            canvas = compose(frame, t_panel, lines, score)
            cv2.imshow("Trainer & User session  (q quit, r reset)", canvas)
            if args.record:
                if writer is None:
                    rec_dir = root / "database" / "recordings"
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    rec_path = rec_dir / f"{args.exercise}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                                             (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
            if finish_at and time.time() > finish_at:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and counter:
                counter.reset()

    cam.release()
    if trainer_cap is not None:
        trainer_cap.release()
    cv2.destroyAllWindows()

    if writer is not None:
        writer.release()
        print(f"Recording saved to {rec_path}")
    mean_score = float(np.mean(scores)) if scores else 0.0
    dur = time.time() - t_start
    if counter:
        reps_total = total_reps + counter.reps
        result_txt = f"reps {reps_total}" + (f" in {sets_done}/{args.sets} sets" if workout else "")
    else:
        reps_total, result_txt = round(hold_time), f"hold {hold_time:.0f}s"
    print(f"\nSession summary: {args.exercise}, {dur:.0f}s, {result_txt}, mean score {mean_score*100:.0f}% ({grade(mean_score)})")
    log = root / "database" / "session_log.csv"
    log.parent.mkdir(exist_ok=True)
    append_log(log, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "exercise": args.exercise,
                     "duration_s": round(dur), "reps_or_hold": reps_total, "mean_score": round(mean_score, 3),
                     "sets_done": sets_done if workout else "", "target_reps": args.target_reps if workout else ""})
    print(f"Logged to {log}")


if __name__ == "__main__":
    main()
