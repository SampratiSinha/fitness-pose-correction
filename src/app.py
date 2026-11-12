"""
app.py - Integrated Fitness Pose-Correction Platform (one dashboard for the whole project)

Pages (sidebar):
  Overview            pipeline status + headline results
  Exercise Guide      how to perform each exercise, what is checked, and how the score works
  Reference Database  images per exercise, angle profiles, rep thresholds, skeleton overlays
  HPE Comparison      MediaPipe vs OpenPose (detection, speed, PCK@0.2), overlay images
  Evaluation          form-scoring results on correct vs incorrect photos
  Trainer & User      Trainer reference view, live session (workout mode, recording), video analysis, snapshot check

Everything is read from the files the other scripts produce in ../database, so
run the pipeline first (python main.py all), then:   streamlit run app.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

from build_database_images import compute_angles, primary_angle, EXERCISES
from comparison_engine import PoseScorer
from guidance import EXERCISE_GUIDE, SCORE_EXPLANATION
from video_analysis import analyze_video

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DB = ROOT / "database"
PROFILES_PATH = DB / "exercise_profiles.json"
HPE_DIR = DB / "hpe_comparison"
EVAL_DIR = DB / "evaluation"
SESSION_LOG = DB / "session_log.csv"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

st.set_page_config(page_title="Fitness Pose Platform", layout="wide")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def not_ready(command):
    st.info(f"Not generated yet. Run `{command}` from the src folder, then reload this page.")


@st.cache_resource
def get_scorer(mtime):
    return PoseScorer(PROFILES_PATH)


def process_snapshot(file_bytes, exercise):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                      min_detection_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None, None, img
    mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    lms = res.pose_landmarks.landmark
    lm_px = np.array([[l.x * w, l.y * h] for l in lms])
    vis = np.array([l.visibility for l in lms])
    angles = compute_angles(lm_px, vis)
    return angles, primary_angle(angles, EXERCISES[exercise]), img


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_overview():
    st.title("Fitness Pose-Correction Platform")
    st.caption("Human pose estimation with MediaPipe: Trainer and User self-practice, validated against OpenPose.")
    st.markdown("**Pipeline:** Literature study → MediaPipe pose extraction → Reference database → "
                "HPE comparison → Scoring engine → Live Trainer & User session")

    parts = [("Reference images", ROOT / "data", "python main.py build"),
             ("Pose database (CSV)", DB / "pose_database.csv", "python main.py build"),
             ("Exercise profiles (JSON)", PROFILES_PATH, "python main.py build"),
             ("HPE comparison results", HPE_DIR / "summary.json", "python main.py compare"),
             ("Scoring evaluation", EVAL_DIR / "summary.json", "python main.py evaluate"),
             ("Live session log", SESSION_LOG, "python main.py live --exercise squat")]
    st.subheader("Pipeline status")
    st.dataframe(pd.DataFrame([{"Component": n, "Status": "Ready" if p.exists() else "Missing", "Command": c}
                               for n, p, c in parts]))

    hpe, ev, sessions = load_json(HPE_DIR / "summary.json"), load_json(EVAL_DIR / "summary.json"), load_csv(SESSION_LOG)
    st.subheader("Headline results")
    cols = st.columns(4)
    if hpe:
        cols[0].metric("MediaPipe speed vs OpenPose", f"{hpe['speedup_mediapipe_vs_openpose']}x faster")
        cols[1].metric("Agreement with OpenPose (PCK@0.2)", f"{hpe['overall_pck'] * 100:.1f}%")
    if ev:
        cols[2].metric("Form-scoring accuracy", f"{ev['overall_accuracy'] * 100:.1f}%")
    if sessions is not None:
        cols[3].metric("Live sessions logged", len(sessions))
    if not (hpe or ev or sessions is not None):
        st.write("Results will appear here once the pipeline has been run.")


def page_guide():
    st.header("Exercise Guide and Scoring")
    exercise = st.selectbox("Exercise", list(EXERCISE_GUIDE))
    g = EXERCISE_GUIDE[exercise]
    st.subheader(g["title"])
    st.write(f"**Camera:** {g['view']}  \n**Setup:** {g['setup']}")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Steps**")
    for i, step in enumerate(g["steps"], 1):
        c1.write(f"{i}. {step}")
    c2.markdown("**Key points**")
    for item in g["key_points"]:
        c2.write(f"- {item}")
    c3.markdown("**Common mistakes**")
    for item in g["mistakes"]:
        c3.write(f"- {item}")
    st.info(f"**What the system checks:** {g['checks']}")
    st.divider()
    st.subheader("How the score works")
    st.markdown(SCORE_EXPLANATION)


def show_video_report(r):
    c = st.columns(4)
    if r["reps"] is not None:
        c[0].metric("Reps counted", r["reps"])
    else:
        c[0].metric("Hold time (good form)", f"{r['hold_s']} s")
    c[1].metric(f"Mean score ({r['grade']})", f"{r['mean_score'] * 100:.0f}%")
    c[2].metric("Frames with good form", f"{r['good_fraction'] * 100:.0f}%")
    c[3].metric("Video length", f"{r['duration_s']} s")
    st.subheader("Score over time")
    st.line_chart(r["series"].set_index("time_s")["score"])
    if r["top_cues"]:
        st.subheader("Most frequent coaching cues")
        for cue, _, frac in r["top_cues"]:
            st.write(f"- **{frac * 100:.0f}% of frames:** {cue}")
    else:
        st.success("No major form issues were flagged.")
    left, right = st.columns(2)
    left.image(r["best"]["image"], caption=f"Best moment: {r['best']['score'] * 100:.0f}% at {r['best']['time_s']} s")
    right.image(r["worst"]["image"], caption=f"Weakest moment: {r['worst']['score'] * 100:.0f}% at {r['worst']['time_s']} s")


def page_database():
    st.header("Reference Database")
    prof, df = load_json(PROFILES_PATH), load_csv(DB / "pose_database.csv")
    if prof is None or df is None:
        return not_ready("python main.py build")

    left, right = st.columns([1, 2])
    left.subheader("Images per exercise")
    left.dataframe(df.groupby(["exercise", "label"]).size().unstack(fill_value=0))

    rows = []
    for ex, p in prof.items():
        for ph, stats in p["phases"].items():
            for joint, s in stats.items():
                rows.append({"Exercise": ex, "Phase": ph, "Joint": joint, "Mean angle (deg)": round(s["mean"], 1),
                             "Tolerance (deg)": round(s.get("suggested_tolerance_deg", 10), 1), "n": s["n"]})
    right.subheader("Reference angle profiles (correct form)")
    right.dataframe(pd.DataFrame(rows))

    thr = [{"Exercise": ex, "Low (deg)": round(p["rep_thresholds"]["low"], 1),
            "High (deg)": round(p["rep_thresholds"]["high"], 1)} for ex, p in prof.items() if "rep_thresholds" in p]
    if thr:
        st.subheader("Rep-counting thresholds (derived from the reference images)")
        st.dataframe(pd.DataFrame(thr))

    st.subheader("Database rows (angles only; the 132 keypoint columns are hidden)")
    st.dataframe(df[[c for c in df.columns if not c.startswith("kp")]])

    shots = sorted((DB / "debug").glob("*.jpg")) if (DB / "debug").exists() else []
    if shots:
        st.subheader("Detected skeletons")
        pick = st.selectbox("Image", [p.name for p in shots])
        st.image(str(DB / "debug" / pick))


def page_hpe():
    st.header("HPE Comparison: MediaPipe vs OpenPose")
    st.caption("Same image dataset for both methods. OpenPose is the reference (not ground truth); "
               "agreement is PCK@0.2: a keypoint counts as agreeing if it lies within 20% of torso length.")
    s = load_json(HPE_DIR / "summary.json")
    if s is None:
        return not_ready("python main.py compare")

    c = st.columns(4)
    c[0].metric("MediaPipe detection rate", f"{s['mediapipe_detection_rate'] * 100:.0f}%")
    c[1].metric("OpenPose detection rate", f"{s['openpose_detection_rate'] * 100:.0f}%")
    c[2].metric("Speed", f"{s['speedup_mediapipe_vs_openpose']}x faster")
    c[3].metric("Overall PCK@0.2", f"{s['overall_pck'] * 100:.1f}%")
    st.write(f"Mean inference time per image: MediaPipe **{s['mediapipe_mean_inference_ms']} ms**, "
             f"OpenPose **{s['openpose_mean_inference_ms']} ms** (CPU only).")

    st.subheader("Per-joint agreement (PCK@0.2)")
    st.bar_chart(pd.Series(s["per_joint_pck"]).sort_values(ascending=False))

    overlays = sorted((HPE_DIR / "overlays").glob("*.jpg")) if (HPE_DIR / "overlays").exists() else []
    if overlays:
        st.subheader("Side-by-side overlays (green = MediaPipe, red = OpenPose)")
        for i in range(0, len(overlays), 2):
            for col, p in zip(st.columns(2), overlays[i:i + 2]):
                col.image(str(p), caption=p.stem)


def page_eval():
    st.header("Evaluation: form scoring on correct vs incorrect photos")
    s, scores = load_json(EVAL_DIR / "summary.json"), load_csv(EVAL_DIR / "scores.csv")
    if s is None or scores is None:
        return not_ready("python main.py evaluate")
    if Path(str(s.get("test_dir", ""))).name == "data":
        st.warning("These results are in-sample (the test images were also used to build the reference profiles), "
                   "so they are optimistic. Add held-out photos to the test folder and re-run for the real result.")

    c = st.columns(4)
    c[0].metric("Images scored", s["n_images"])
    c[1].metric("Accuracy", f"{s['overall_accuracy'] * 100:.1f}%")
    if s.get("mean_score_correct") is not None:
        c[2].metric("Mean score, correct form", f"{s['mean_score_correct'] * 100:.0f}%")
    if s.get("mean_score_incorrect") is not None:
        c[3].metric("Mean score, incorrect form", f"{s['mean_score_incorrect'] * 100:.0f}%")

    left, right = st.columns(2)
    left.subheader("Outcomes")
    left.dataframe(pd.DataFrame(list(s["confusion"].items()), columns=["Outcome", "Count"]))
    per_ex = pd.DataFrame(s["per_exercise"]).T
    right.subheader("Per exercise")
    right.dataframe(per_ex)
    st.subheader("Mean form score by exercise")
    st.bar_chart(per_ex[["mean_correct", "mean_incorrect"]])
    st.subheader("Every scored image")
    st.dataframe(scores.drop(columns=["hit"], errors="ignore"))


def page_trainer_user():
    st.header("Trainer & User")
    if not PROFILES_PATH.exists():
        return not_ready("python main.py build")
    scorer = get_scorer(PROFILES_PATH.stat().st_mtime)
    exercise = st.selectbox("Exercise", list(scorer.profiles.keys()))
    role = st.radio("I am a:", ["User (self-practice)", "Trainer (view reference)"], horizontal=True)

    if role.startswith("Trainer"):
        prof = scorer.profiles[exercise]
        st.write(f"**Primary joint:** {prof['primary_joint']}  |  **Static hold:** {prof['static']}")
        for phase, stats in prof["phases"].items():
            st.subheader(f"Phase: {phase}")
            for col, (joint, s) in zip(st.columns(len(stats) or 1), stats.items()):
                col.metric(joint, f"{s['mean']:.0f} deg", f"±{s.get('suggested_tolerance_deg', 10):.0f} deg")
        st.caption("This is the reference the User view scores against.")
        return

    st.subheader("Live session (video + rep counting)")
    st.write("Opens your webcam next to the trainer. Press **q** in the video window to finish; **r** resets the rep count.")
    c1, c2 = st.columns([3, 1])
    trainer_video = c1.text_input("Trainer video path (optional)", placeholder="../trainer/squat.mp4")
    cam_idx = c2.number_input("Camera index", min_value=0, max_value=5, value=0)
    w1, w2, w3 = st.columns(3)
    target = w1.number_input("Target reps per set (0 = free practice)", min_value=0, max_value=100, value=0)
    sets = w2.number_input("Sets", min_value=1, max_value=10, value=1)
    record = w3.checkbox("Record the session video")
    if st.button("Start live session"):
        cmd = [sys.executable, str(SRC / "live_session.py"), "--exercise", exercise, "--camera", str(int(cam_idx))]
        if trainer_video.strip():
            cmd += ["--trainer-video", trainer_video.strip()]
        if target > 0:
            cmd += ["--target-reps", str(int(target)), "--sets", str(int(sets))]
        if record:
            cmd += ["--record"]
        subprocess.Popen(cmd, cwd=str(SRC))
        st.success("Live session started in a separate window.")

    history = load_csv(SESSION_LOG)
    if history is not None:
        st.caption("Session history")
        st.dataframe(history.tail(10))

    st.divider()
    st.subheader("Analyze a recorded video")
    st.write("Upload a short side-view clip of your set. The whole video is scored, reps are counted, "
             "and your best and weakest moments are shown.")
    clip = st.file_uploader("Exercise video", type=["mp4", "mov", "avi", "mkv"])
    if clip is not None and st.button("Analyze video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(clip.name).suffix or ".mp4") as tmp:
            tmp.write(clip.getvalue())
            tmp_path = tmp.name
        bar = st.progress(0.0)
        report = None
        try:
            report = analyze_video(tmp_path, exercise, scorer, progress=bar.progress)
        except ValueError as err:
            st.error(str(err))
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        if report:
            show_video_report(report)

    st.divider()
    st.subheader("Snapshot check")
    st.write("Position yourself side-on to the camera, full body visible, then take a snapshot.")
    shot = st.camera_input("Your snapshot")
    if shot is not None:
        angles, pa, annotated = process_snapshot(shot.getvalue(), exercise)
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected pose")
        if angles is None:
            col2.error("No person detected. Make sure your full body is visible and well lit.")
        else:
            result = scorer.score(angles, exercise, primary_angle=pa)
            col2.metric("Phase detected", result["phase"])
            if result["angle_score"] is not None:
                col2.metric(f"Form score ({result['grade']})", f"{result['angle_score'] * 100:.0f}%")
            col2.subheader("Coaching")
            for line in result["feedback"]:
                col2.write(f"- {line}")
            if result["per_joint"]:
                col2.subheader("Per-joint breakdown")
                col2.dataframe(pd.DataFrame(result["per_joint"]).T)


PAGES = {"Overview": page_overview, "Exercise Guide": page_guide, "Reference Database": page_database, "HPE Comparison": page_hpe,
         "Evaluation": page_eval, "Trainer & User": page_trainer_user}
PAGES[st.sidebar.radio("Go to", list(PAGES))]()
