"""
main.py - single entry point for the whole project (run from src/)

    python main.py                  interactive menu
    python main.py status           what has been generated so far
    python main.py demo             1. MediaPipe live webcam demo
    python main.py build            2. build the reference database from data/
    python main.py compare          3. MediaPipe vs OpenPose comparison
    python main.py evaluate         4. score correct vs incorrect photos
    python main.py live --exercise squat [--trainer-video ../trainer/squat.mp4]
                        [--target-reps 10 --sets 3] [--record]      workout mode, session recording
    python main.py analyze --video ../my_set.mp4 --exercise squat    score a recorded video
    python main.py dashboard        6. open the integrated Streamlit dashboard
    python main.py all              run build, compare, evaluate in order

Add --dry-run (before the command) to print what would run without running it.
Each step checks its prerequisites first and tells you what is missing.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DATA, TEST, DB = ROOT / "data", ROOT / "test", ROOT / "database"
OPENPOSE_DIR = ROOT / "models" / "openpose"
OPENPOSE_FILES = ["pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel"]
PROFILES = DB / "exercise_profiles.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DRY_RUN = False


def count_images(folder):
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT) if folder.exists() else 0


def run(args, label):
    cmd = [sys.executable, *args]
    print(f"\n>>> {label}\n    $ {' '.join(cmd)}")
    return 0 if DRY_RUN else subprocess.call(cmd, cwd=str(SRC))


def fail(*messages):
    for m in messages:
        print(f"  ! {m}")
    return 1


def profile_exercises():
    try:
        with open(PROFILES) as f:
            return list(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def step_demo():
    return run(["pose_extraction.py"], "1. MediaPipe webcam demo (press q in the window to quit)")


def step_build():
    if count_images(DATA) == 0:
        return fail(f"No images found in {DATA}.", "Put photos in data/<exercise>/<correct|incorrect>/ first.")
    return run(["build_database_images.py"], "2. Build the reference database")


def step_compare():
    if count_images(DATA) == 0:
        return fail(f"No images found in {DATA}.")
    missing = [f for f in OPENPOSE_FILES if not (OPENPOSE_DIR / f).exists()]
    if missing:
        return fail(f"OpenPose model files missing in {OPENPOSE_DIR}: {', '.join(missing)}",
                    "See the download commands in the README.")
    return run(["compare_hpe_methods.py", "--openpose-dir", str(OPENPOSE_DIR)], "3. MediaPipe vs OpenPose comparison")


def step_evaluate(test_dir=None):
    if not PROFILES.exists():
        return fail("No reference profiles yet. Run: python main.py build")
    folder = Path(test_dir) if test_dir else (TEST if count_images(TEST) else DATA)
    if folder == DATA:
        print("  Note: no held-out photos in test/, so this evaluation is in-sample (optimistic).")
    return run(["evaluate_scoring.py", "--test-dir", str(folder)], f"4. Evaluate form scoring on {folder.name}/")


def step_live(exercise, trainer_video=None, camera=0, target_reps=0, sets=1, record=False):
    known = profile_exercises()
    if not known:
        return fail("No reference profiles yet. Run: python main.py build")
    if exercise not in known:
        return fail(f"'{exercise}' is not in the database. Available: {', '.join(known)}")
    args = ["live_session.py", "--exercise", exercise, "--camera", str(camera)]
    if trainer_video:
        args += ["--trainer-video", str(trainer_video)]
    if target_reps and target_reps > 0:
        args += ["--target-reps", str(target_reps), "--sets", str(sets)]
    if record:
        args += ["--record"]
    return run(args, f"5. Live Trainer & User session: {exercise} (q quit, r reset reps)")


def step_analyze(video, exercise):
    if not PROFILES.exists():
        return fail("No reference profiles yet. Run: python main.py build")
    if exercise not in profile_exercises():
        return fail(f"'{exercise}' is not in the database. Available: {', '.join(profile_exercises())}")
    if not Path(video).exists():
        return fail(f"Video not found: {video}")
    return run(["video_analysis.py", "--video", str(video), "--exercise", exercise], f"7. Analyze a recorded video: {exercise}")


def step_dashboard():
    if importlib.util.find_spec("streamlit") is None:
        return fail("Streamlit is not installed. Run: pip install -r requirements.txt")
    return run(["-m", "streamlit", "run", "app.py"], "6. Integrated dashboard")


def step_status():
    print("\nProject status")
    print("-" * 64)
    items = [("Reference images (data/)", f"{count_images(DATA)} images", count_images(DATA) > 0),
             ("Held-out test photos (test/)", f"{count_images(TEST)} images", count_images(TEST) > 0),
             ("OpenPose model files", str(OPENPOSE_DIR), all((OPENPOSE_DIR / f).exists() for f in OPENPOSE_FILES)),
             ("Pose database", "database/pose_database.csv", (DB / "pose_database.csv").exists()),
             ("Exercise profiles", "database/exercise_profiles.json", PROFILES.exists()),
             ("HPE comparison", "database/hpe_comparison/summary.json", (DB / "hpe_comparison" / "summary.json").exists()),
             ("Scoring evaluation", "database/evaluation/summary.json", (DB / "evaluation" / "summary.json").exists()),
             ("Live session log", "database/session_log.csv", (DB / "session_log.csv").exists())]
    for name, detail, ok in items:
        print(f"  [{'OK' if ok else '--'}] {name:<30} {detail}")
    for path, label, keys in ((DB / "hpe_comparison" / "summary.json", "HPE comparison",
                               ("speedup_mediapipe_vs_openpose", "overall_pck")),
                              (DB / "evaluation" / "summary.json", "Evaluation", ("overall_accuracy",))):
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                print(f"  {label}: " + ", ".join(f"{k}={data[k]}" for k in keys))
            except (json.JSONDecodeError, KeyError):
                pass
    exercises = profile_exercises()
    if exercises:
        print(f"  Exercises in database: {', '.join(exercises)}")
    return 0


def step_all():
    for fn in (step_build, step_compare, step_evaluate):
        code = fn()
        if code != 0:
            if fn is step_compare:
                print("  (skipping the OpenPose comparison and continuing)")
                continue
            print("\nStopped: fix the problem above and run again.")
            return code
    return step_status()


# ---------------------------------------------------------------------------
# menu + CLI
# ---------------------------------------------------------------------------
def menu():
    options = [("1", "Webcam pose demo (check MediaPipe works)", step_demo),
               ("2", "Build reference database from images", step_build),
               ("3", "Compare MediaPipe vs OpenPose", step_compare),
               ("4", "Evaluate form scoring", step_evaluate),
               ("5", "Live Trainer & User session", None),
               ("6", "Open integrated dashboard", step_dashboard),
               ("7", "Project status", step_status),
               ("8", "Run steps 2, 3 and 4 in order", step_all),
               ("9", "Analyze a recorded video", None)]
    while True:
        print("\n=== Fitness Pose-Correction Project ===")
        for key, label, _ in options:
            print(f"  {key}  {label}")
        print("  0  Exit")
        choice = input("Choose: ").strip()
        if choice == "0":
            return 0
        if choice == "5":
            names = profile_exercises()
            if not names:
                fail("No reference profiles yet. Choose option 2 first.")
                continue
            for i, n in enumerate(names, 1):
                print(f"  {i}  {n}")
            pick = input("Exercise number: ").strip()
            if not pick.isdigit() or not 1 <= int(pick) <= len(names):
                print("  ! Not a valid choice.")
                continue
            video = input("Trainer video path (blank = reference image): ").strip()
            reps = input("Target reps per set (blank = free practice): ").strip()
            sets = input("Number of sets: ").strip() if reps.isdigit() and int(reps) > 0 else "1"
            rec = input("Record the session video? (y/N): ").strip().lower() == "y"
            step_live(names[int(pick) - 1], video or None, 0, int(reps) if reps.isdigit() else 0,
                      int(sets) if sets.isdigit() and int(sets) > 0 else 1, rec)
            continue
        if choice == "9":
            names = profile_exercises()
            if not names:
                fail("No reference profiles yet. Choose option 2 first.")
                continue
            print("  Exercises: " + ", ".join(names))
            step_analyze(input("Video path: ").strip(), input("Exercise: ").strip())
            continue
        match = [fn for key, _, fn in options if key == choice]
        if match:
            match[0]()
        else:
            print("  ! Not a valid choice.")


def main():
    global DRY_RUN
    ap = argparse.ArgumentParser(description="Fitness pose-correction project launcher")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("demo", "build", "compare", "dashboard", "status", "all"):
        sub.add_parser(name)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--test-dir")
    lv = sub.add_parser("live")
    lv.add_argument("--exercise", required=True)
    lv.add_argument("--trainer-video")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--target-reps", type=int, default=0)
    lv.add_argument("--sets", type=int, default=1)
    lv.add_argument("--record", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("--video", required=True)
    an.add_argument("--exercise", required=True)
    args = ap.parse_args()
    DRY_RUN = args.dry_run

    if args.cmd is None:
        return menu()
    if args.cmd == "evaluate":
        return step_evaluate(args.test_dir)
    if args.cmd == "live":
        return step_live(args.exercise, args.trainer_video, args.camera, args.target_reps, args.sets, args.record)
    if args.cmd == "analyze":
        return step_analyze(args.video, args.exercise)
    return {"demo": step_demo, "build": step_build, "compare": step_compare, "dashboard": step_dashboard,
            "status": step_status, "all": step_all}[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
