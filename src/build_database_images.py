import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

ANGLE_DEFS = {
    "elbow":     ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)),
    "knee":      ((L_HIP, L_KNEE, L_ANKLE),       (R_HIP, R_KNEE, R_ANKLE)),
    "hip":       ((L_SHOULDER, L_HIP, L_KNEE),    (R_SHOULDER, R_HIP, R_KNEE)),
    "body_line": ((L_SHOULDER, L_HIP, L_ANKLE),   (R_SHOULDER, R_HIP, R_ANKLE)),
}
ALL_JOINTS = list(ANGLE_DEFS) + ["torso_lean"]

EXERCISES = {
    "squat": {
        "primary_joint": "knee", "primary_mode": "best_side",
        "high_phase": "up", "low_phase": "down", "static": False,
        "key_angles": ["knee", "hip", "torso_lean"],
    },
    "pushup": {
        "primary_joint": "elbow", "primary_mode": "best_side",
        "high_phase": "up", "low_phase": "down", "static": False,
        "key_angles": ["elbow", "body_line"],
    },
    "bicep_curl": {
        "primary_joint": "elbow", "primary_mode": "best_side",
        "high_phase": "extended", "low_phase": "curled", "static": False,
        "key_angles": ["elbow", "torso_lean"],
    },
    "shoulder_press": {
        "primary_joint": "elbow", "primary_mode": "best_side",
        "high_phase": "up", "low_phase": "down", "static": False,
        "key_angles": ["elbow", "torso_lean"],
    },
    "plank": {
        "primary_joint": "body_line", "primary_mode": "best_side",
        "high_phase": "hold", "low_phase": "hold", "static": True,
        "key_angles": ["body_line", "hip"],
    },
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_TOLERANCE_DEG = 10.0

def angle_at(a, b, c):
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1, 1))))

def torso_lean_angle(hip, shoulder):
    v = shoulder - hip
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(-v[1] / n, -1, 1))))

def normalize_keypoints(lm_px, z, vis):
    hip_c = (lm_px[L_HIP] + lm_px[R_HIP]) / 2
    sh_c = (lm_px[L_SHOULDER] + lm_px[R_SHOULDER]) / 2
    torso = np.linalg.norm(sh_c - hip_c)
    if torso < 1e-6:
        return None
    return np.column_stack([(lm_px - hip_c) / torso, z / torso, vis])

def compute_angles(lm_px, vis):
    out = {}
    for joint, triplets in ANGLE_DEFS.items():
        vals, viss = [], []
        for t in triplets:
            vals.append(angle_at(lm_px[t[0]], lm_px[t[1]], lm_px[t[2]]))
            viss.append(min(vis[t[0]], vis[t[1]], vis[t[2]]))
        out[f"{joint}_l"], out[f"{joint}_r"] = vals
        best = 0 if viss[0] >= viss[1] else 1
        out[joint] = vals[best] if not np.isnan(vals[best]) else vals[1 - best]

    vals, viss = [], []
    for hi, si in ((L_HIP, L_SHOULDER), (R_HIP, R_SHOULDER)):
        vals.append(torso_lean_angle(lm_px[hi], lm_px[si]))
        viss.append(min(vis[hi], vis[si]))
    out["torso_lean_l"], out["torso_lean_r"] = vals
    best = 0 if viss[0] >= viss[1] else 1
    out["torso_lean"] = vals[best] if not np.isnan(vals[best]) else vals[1 - best]
    return out

def primary_angle(row, cfg):
    j = cfg["primary_joint"]
    if cfg["primary_mode"] == "min":
        v = [x for x in (row[f"{j}_l"], row[f"{j}_r"]) if not np.isnan(x)]
        return min(v) if v else np.nan
    return row[j]

def extract_row(pose, img_bgr, min_vis):
    h, w = img_bgr.shape[:2]
    results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, "no person detected"
    lms = results.pose_landmarks.landmark
    lm_px = np.array([[l.x * w, l.y * h] for l in lms], dtype=float)
    z = np.array([l.z * w for l in lms], dtype=float)
    vis = np.array([l.visibility for l in lms], dtype=float)
    if vis[BODY_LANDMARKS].mean() < min_vis:
        return None, "body not fully visible (low confidence)"
    kp = normalize_keypoints(lm_px, z, vis)
    if kp is None:
        return None, "degenerate pose"
    row = compute_angles(lm_px, vis)
    for i in range(33):
        row[f"kp{i}_x"], row[f"kp{i}_y"], row[f"kp{i}_z"], row[f"kp{i}_v"] = kp[i]
    return row, results

def _canon(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def match_exercise(folder):
    c = _canon(folder)
    aliases = {"pushups": "pushup", "bicepcurls": "bicep_curl", "lunges": "lunge",
               "squats": "squat", "planks": "plank",
               "shoulderpresses": "shoulder_press"}
    for k in EXERCISES:
        if _canon(k) == c:
            return k
    return aliases.get(c)

def collect_images(data_dir):
    items = []
    for ex_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        ex = match_exercise(ex_dir.name)
        if ex is None:
            print(f"[skip] '{ex_dir.name}' is not a configured exercise "
                  f"({', '.join(EXERCISES)})")
            continue
        cfg = EXERCISES[ex]
        for f in sorted(ex_dir.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXT:
                continue
            parts = [p.lower() for p in f.relative_to(ex_dir).parts[:-1]]
            label = "incorrect" if "incorrect" in parts else "correct"
            folder_phase = next((p for p in parts
                                 if p in (cfg["high_phase"], cfg["low_phase"])
                                 and not cfg["static"]), None)
            items.append((ex, label, folder_phase, f))
    return items

def stats_for(g, key_angles, with_tolerance=False):
    out = {}
    for a in key_angles:
        s = g[a].dropna()
        if len(s):
            d = {"mean": float(s.mean()), "std": float(s.std(ddof=0)),
                 "min": float(s.min()), "max": float(s.max()), "n": int(len(s))}
            if with_tolerance:
                d["suggested_tolerance_deg"] = float(max(2 * d["std"], MIN_TOLERANCE_DEG))
            out[a] = d
    return out

def label_and_profile(df):
    df = df.copy()
    df["primary_angle"] = np.nan
    df["phase"], df["phase_source"] = "", ""
    profiles, warnings = {}, []

    for ex, cfg in EXERCISES.items():
        m = df["exercise"] == ex
        if not m.any():
            continue
        df.loc[m, "primary_angle"] = df[m].apply(lambda r: primary_angle(r, cfg), axis=1)
        for _, r in df[m & df["primary_angle"].isna()].iterrows():
            warnings.append(f"{r['image']}: could not measure the {cfg['primary_joint']} angle "
                            f"(joint hidden?) - replace this photo.")
        df = df.drop(index=df.index[m & df["primary_angle"].isna()])
        m = df["exercise"] == ex
        if not m.any():
            continue
        correct = df[m & (df["label"] == "correct")]
        prof = {"view": "side", "primary_joint": cfg["primary_joint"],
                "primary_mode": cfg["primary_mode"], "static": cfg["static"],
                "n_correct": int(len(correct)),
                "n_incorrect": int((m & (df["label"] == "incorrect")).sum()),
                "phases": {}, "incorrect": {}}

        if cfg["static"]:
            df.loc[m, "phase"], df.loc[m, "phase_source"] = "hold", "static"
        else:
            hi_imgs = correct[correct["phase_folder"] == cfg["high_phase"]]["primary_angle"].dropna()
            lo_imgs = correct[correct["phase_folder"] == cfg["low_phase"]]["primary_angle"].dropna()
            if len(hi_imgs) and len(lo_imgs):
                hi_mean, lo_mean = hi_imgs.median(), lo_imgs.median()  # median: robust to one bad photo
            else:
                s = correct["primary_angle"].dropna()
                if len(s) < 2:
                    warnings.append(f"{ex}: not enough correct images to derive phases.")
                    continue
                hi_mean, lo_mean = s.max(), s.min()
                warnings.append(f"{ex}: put correct images into '{cfg['high_phase']}/' and "
                                f"'{cfg['low_phase']}/' folders for reliable phases "
                                f"(fell back to min/max of correct images).")
            if hi_mean <= lo_mean:
                warnings.append(f"{ex}: '{cfg['high_phase']}' images have a SMALLER "
                                f"{cfg['primary_joint']} angle than '{cfg['low_phase']}' "
                                f"images - folders look swapped or a pose was mis-detected.")
            rng = hi_mean - lo_mean
            split = (hi_mean + lo_mean) / 2
            prof["primary_high_mean"], prof["primary_low_mean"] = float(hi_mean), float(lo_mean)
            prof["phase_split_angle"] = float(split)
            prof["rep_thresholds"] = {"low": float(lo_mean + 0.35 * rng),
                                      "high": float(lo_mean + 0.65 * rng)}

            for idx in df.index[m]:
                fp = df.at[idx, "phase_folder"]
                if fp:
                    df.at[idx, "phase"], df.at[idx, "phase_source"] = fp, "folder"
                    ang = df.at[idx, "primary_angle"]
                    wrong = (fp == cfg["high_phase"] and ang < split) or \
                            (fp == cfg["low_phase"] and ang >= split)
                    if wrong and df.at[idx, "label"] == "correct" and not np.isnan(ang):
                        warnings.append(f"{df.at[idx, 'image']}: filed under '{fp}' but its "
                                        f"{cfg['primary_joint']} angle ({ang:.0f} deg) looks like "
                                        f"the other phase - check the photo.")
                else:
                    ang = df.at[idx, "primary_angle"]
                    df.at[idx, "phase"] = cfg["high_phase"] if ang >= split else cfg["low_phase"]
                    df.at[idx, "phase_source"] = "auto"

        correct = df[m & (df["label"] == "correct")]
        incorrect = df[m & (df["label"] == "incorrect")]
        for ph in sorted(set(df.loc[m, "phase"])):
            c = correct[correct["phase"] == ph]
            i = incorrect[incorrect["phase"] == ph]
            if len(c):
                prof["phases"][ph] = stats_for(c, cfg["key_angles"], with_tolerance=True)
            if len(i):
                prof["incorrect"][ph] = stats_for(i, cfg["key_angles"])
        profiles[ex] = prof
    return df, profiles, warnings

def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Build the fitness pose database from images.")
    ap.add_argument("--data", type=Path, default=root / "data")
    ap.add_argument("--out", type=Path, default=root / "database")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2],
                    help="MediaPipe model (2=heavy, most accurate; fine for a few images)")
    ap.add_argument("--min-vis", type=float, default=0.5)
    ap.add_argument("--no-debug", action="store_true", help="skip skeleton overlay images")
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"Data folder not found: {args.data}")
    items = collect_images(args.data)
    if not items:
        sys.exit("No images found. Expected data/<exercise>/<correct|incorrect>/... "
                 "(see the docstring at the top of this file).")
    args.out.mkdir(parents=True, exist_ok=True)
    debug_dir = args.out / "debug"
    if not args.no_debug:
        debug_dir.mkdir(exist_ok=True)

    print(f"Found {len(items)} image(s). Extracting poses...\n")
    rows, failed, overlays = [], [], {}
    with mp_pose.Pose(static_image_mode=True, model_complexity=args.complexity,
                      min_detection_confidence=0.5) as pose:
        for ex, label, folder_phase, path in items:
            rel = str(path.relative_to(args.data)).replace("\\", "/")
            img = cv2.imread(str(path))
            if img is None:
                failed.append((rel, "unreadable file"))
                continue
            row, res = extract_row(pose, img, args.min_vis)
            if row is None:
                failed.append((rel, res))
                continue
            row.update(exercise=ex, label=label, phase_folder=folder_phase or "", image=rel)
            rows.append(row)
            if not args.no_debug:
                vis_img = img.copy()
                mp_draw.draw_landmarks(vis_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                overlays[rel] = vis_img

    if failed:
        print("=== Images that FAILED (replace these with clearer photos) ===")
        for rel, why in failed:
            print(f"  {rel}: {why}")
        print()
    if not rows:
        sys.exit("No usable images. Use full-body, well-lit, side-view photos.")

    df = pd.DataFrame(rows)
    df, profiles, warnings = label_and_profile(df)

    angle_cols = [c for j in ALL_JOINTS for c in (j, f"{j}_l", f"{j}_r")]
    kp_cols = [f"kp{i}_{k}" for i in range(33) for k in "xyzv"]
    meta = ["exercise", "label", "phase", "phase_source", "image"]
    df = df[meta + angle_cols + ["primary_angle"] + kp_cols]

    df.to_csv(args.out / "pose_database.csv", index=False)
    with open(args.out / "exercise_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    if not args.no_debug:
        for _, r in df.iterrows():
            im = overlays[r["image"]]
            txt = f"{r['exercise']} | {r['label']} | {r['phase']} | primary {r['primary_angle']:.0f} deg"
            cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(im, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            name = f"{r['exercise']}__{r['label']}__{Path(r['image']).stem}.jpg"
            cv2.imwrite(str(debug_dir / name), im)

    print("=== Per-image results ===")
    show = df[["exercise", "label", "phase", "primary_angle", "image"]].copy()
    show["primary_angle"] = show["primary_angle"].round(1)
    print(show.to_string(index=False))

    print("\n=== Image counts (target: ~5-6 per exercise, 25-30 total) ===")
    print(df.groupby(["exercise", "label"]).size().unstack(fill_value=0).to_string())
    print(f"Total usable images: {len(df)}")

    if warnings:
        print("\n=== Warnings (please review) ===")
        for w in warnings:
            print(f"  ! {w}")

    print(f"\nSaved: {args.out / 'pose_database.csv'}  ({len(df)} rows)")
    print(f"Saved: {args.out / 'exercise_profiles.json'}")
    if not args.no_debug:
        print(f"Saved: skeleton overlays in {debug_dir}  <- open a few to confirm the skeleton is right")

if __name__ == "__main__":
    main()