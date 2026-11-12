import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MP_IDX = {"nose": 0, "r_shoulder": 12, "l_shoulder": 11, "r_elbow": 14, "l_elbow": 13,
          "r_wrist": 16, "l_wrist": 15, "r_hip": 24, "l_hip": 23, "r_knee": 26,
          "l_knee": 25, "r_ankle": 28, "l_ankle": 27}
OP_IDX = {"nose": 0, "r_shoulder": 2, "l_shoulder": 5, "r_elbow": 3, "l_elbow": 6,
          "r_wrist": 4, "l_wrist": 7, "r_hip": 8, "l_hip": 11, "r_knee": 9,
          "l_knee": 12, "r_ankle": 10, "l_ankle": 13}
COMMON_JOINTS = list(MP_IDX.keys())
N_OP_POINTS = 18

OP_PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0)]

PCK_THRESHOLD = 0.2

def run_mediapipe(pose, img_bgr):
    h, w = img_bgr.shape[:2]
    t0 = time.perf_counter()
    res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dt = time.perf_counter() - t0
    if not res.pose_landmarks:
        return None, dt
    lm = res.pose_landmarks.landmark
    pts = {name: (lm[i].x * w, lm[i].y * h, lm[i].visibility) for name, i in MP_IDX.items()}
    return pts, dt

def load_openpose(proto_path, weights_path):
    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def run_openpose(net, img_bgr, in_size=368, conf_thr=0.1):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0 / 255, (in_size, in_size), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    t0 = time.perf_counter()
    out = net.forward()
    dt = time.perf_counter() - t0

    points = {}
    for i in range(N_OP_POINTS):
        prob_map = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(prob_map)
        if conf > conf_thr:
            x = (w * point[0]) / out.shape[3]
            y = (h * point[1]) / out.shape[2]
            points[i] = (x, y, conf)
    pts = {name: points.get(idx) for name, idx in OP_IDX.items()}
    if sum(v is not None for v in pts.values()) < 4:
        return None, dt
    return pts, dt

def torso_size(op_pts):
    cands = []
    for sh, hip in (("r_shoulder", "l_hip"), ("l_shoulder", "r_hip")):
        if op_pts.get(sh) and op_pts.get(hip):
            a, b = op_pts[sh], op_pts[hip]
            cands.append(np.hypot(a[0] - b[0], a[1] - b[1]))
    return float(np.mean(cands)) if cands else None

def compare_image(mp_pts, op_pts):
    scale = torso_size(op_pts)
    if not scale or scale < 1e-3:
        return None
    rows = []
    for joint in COMMON_JOINTS:
        m, o = mp_pts.get(joint), op_pts.get(joint)
        if m is None or o is None:
            rows.append({"joint": joint, "distance_px": np.nan, "norm_distance": np.nan, "correct": np.nan})
            continue
        d = float(np.hypot(m[0] - o[0], m[1] - o[1]))
        nd = d / scale
        rows.append({"joint": joint, "distance_px": d, "norm_distance": nd, "correct": nd <= PCK_THRESHOLD})
    return rows

def draw_overlay(img, mp_pts, op_pts):
    left = img.copy()
    for name, (x, y, v) in (mp_pts or {}).items():
        if v > 0.3:
            cv2.circle(left, (int(x), int(y)), 4, (0, 255, 0), -1)
    right = img.copy()
    for name, v in (op_pts or {}).items():
        if v:
            cv2.circle(right, (int(v[0]), int(v[1])), 4, (0, 0, 255), -1)
    cv2.putText(left, "MediaPipe", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(right, "OpenPose (reference)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return np.hstack([left, right])

def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=root / "data")
    ap.add_argument("--openpose-dir", type=Path, required=True,
                    help="folder containing pose_deploy_linevec.prototxt and pose_iter_440000.caffemodel")
    ap.add_argument("--out", type=Path, default=root / "database" / "hpe_comparison")
    ap.add_argument("--n-overlays", type=int, default=6)
    args = ap.parse_args()

    proto = args.openpose_dir / "pose_deploy_linevec.prototxt"
    weights = args.openpose_dir / "pose_iter_440000.caffemodel"
    if not proto.exists() or not weights.exists():
        raise SystemExit(f"OpenPose model files not found in {args.openpose_dir}. "
                         f"Expected pose_deploy_linevec.prototxt and pose_iter_440000.caffemodel.")

    images = sorted(p for p in args.data.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not images:
        raise SystemExit(f"No images found under {args.data}")
    print(f"Found {len(images)} images.\n")

    args.out.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    print("Loading OpenPose (OpenCV DNN, COCO model)...")
    net = load_openpose(proto, weights)
    print("Loading MediaPipe Pose...")

    rows, mp_times, op_times, mp_fail, op_fail, saved = [], [], [], 0, 0, 0
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            rel = str(img_path.relative_to(args.data))
            exercise = img_path.relative_to(args.data).parts[0]

            mp_pts, mp_dt = run_mediapipe(pose, img)
            op_pts, op_dt = run_openpose(net, img)
            mp_times.append(mp_dt); op_times.append(op_dt)
            if mp_pts is None: mp_fail += 1
            if op_pts is None: op_fail += 1

            if mp_pts is not None and op_pts is not None:
                cmp_rows = compare_image(mp_pts, op_pts)
                if cmp_rows:
                    for r in cmp_rows:
                        r.update(image=rel, exercise=exercise)
                    rows.extend(cmp_rows)
                if saved < args.n_overlays:
                    cv2.imwrite(str(overlay_dir / f"{saved:02d}_{img_path.stem}.jpg"),
                               draw_overlay(img, mp_pts, op_pts))
                    saved += 1
            print(f"  {rel:<45} MediaPipe {'OK' if mp_pts else 'FAIL':<5} ({mp_dt*1000:5.0f} ms)   "
                 f"OpenPose {'OK' if op_pts else 'FAIL':<5} ({op_dt*1000:5.0f} ms)")

    if not rows:
        raise SystemExit("No images had BOTH methods succeed - cannot compute agreement. "
                         "Check that images are full-body and well lit.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "comparison_results.csv", index=False)

    valid = df.dropna(subset=["correct"])
    per_joint = valid.groupby("joint").agg(
        pck=("correct", "mean"), mean_norm_dist=("norm_distance", "mean"), n=("correct", "size")).round(3)

    summary = {
        "n_images_total": len(images),
        "n_images_both_detected": int(df["image"].nunique()),
        "mediapipe_detection_rate": round(1 - mp_fail / len(images), 3),
        "openpose_detection_rate": round(1 - op_fail / len(images), 3),
        "mediapipe_mean_inference_ms": round(float(np.mean(mp_times)) * 1000, 1),
        "openpose_mean_inference_ms": round(float(np.mean(op_times)) * 1000, 1),
        "speedup_mediapipe_vs_openpose": round(float(np.mean(op_times)) / float(np.mean(mp_times)), 1),
        "pck_threshold": PCK_THRESHOLD,
        "overall_pck": round(float(valid["correct"].mean()), 3),
        "per_joint_pck": per_joint["pck"].to_dict(),
        "mean_normalized_distance": round(float(valid["norm_distance"].mean()), 3),
    }
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Detection rate ===")
    print(f"MediaPipe: {summary['mediapipe_detection_rate']*100:.0f}%   "
         f"OpenPose: {summary['openpose_detection_rate']*100:.0f}%")
    print("\n=== Speed (mean per image) ===")
    print(f"MediaPipe: {summary['mediapipe_mean_inference_ms']} ms   "
         f"OpenPose: {summary['openpose_mean_inference_ms']} ms   "
         f"-> MediaPipe is {summary['speedup_mediapipe_vs_openpose']}x faster")
    print(f"\n=== Agreement with OpenPose (reference), PCK@{PCK_THRESHOLD} ===")
    print(f"Overall: {summary['overall_pck']*100:.1f}%")
    print(per_joint.to_string())
    print(f"\nSaved: {args.out/'comparison_results.csv'}")
    print(f"Saved: {args.out/'summary.json'}")
    print(f"Saved: {saved} overlay images in {overlay_dir}")

if __name__ == "__main__":
    main()