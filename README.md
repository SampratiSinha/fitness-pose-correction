# Fitness Pose-Correction Platform (MediaPipe)

A Trainer and User platform for self-practice. The user follows a trainer (video or reference image) in front of a webcam and gets a live form score, grade, coaching cues, phase detection and repetition counting. MediaPipe Pose provides the landmarks, a reference database of exercise poses provides the targets, and the choice of MediaPipe was validated against OpenPose.

Exercises: squat, push-up, bicep curl, shoulder press, plank.

Author: Your Name | Project review submission

## Features
- **Reference database** from photos: normalised landmarks, joint angles, phase profiles and rep thresholds
- **Score function and grade**: per-joint score from angle deviation, blended with the worst joint (Excellent, Good, Needs work, Poor)
- **Coaching cues**: ranked per joint, phase and direction, with your angle against the reference
- **Live video session** beside the trainer, with rep counting (hold time for plank), workout mode (target reps and sets) and optional recording
- **Video analysis**: score a whole recorded clip, with score over time, top cues, best and weakest moments
- **Integrated dashboard** (Streamlit): pipeline status, database, HPE comparison, evaluation, exercise guide, Trainer & User
- **MediaPipe vs OpenPose study** on the same image set, with OpenPose as the reference

## Results: MediaPipe vs OpenPose (30 images, CPU only)
| Metric | MediaPipe | OpenPose |
|---|---|---|
| Detection rate | 100% (30/30) | 96.7% (29/30) |
| Mean inference time per image | 136.8 ms | 1468.7 ms |
| Speed | 10.7x faster | baseline |
| Agreement with OpenPose, PCK@0.2 | 83.1% | reference |

OpenPose is used as a reference, not as ground truth. Agreement was highest for the nose and wrists (about 90 to 96%) and lowest for the hips and right knee (69 to 74%). Full details are in the survey: [docs/Literature_Survey.pdf](docs/Literature_Survey.pdf), Section 7.

<!-- Add your held-out form-scoring result here, e.g. "Form-scoring accuracy: XX% on N held-out photos (python main.py evaluate)". -->

<!-- Screenshots: add the images to docs/screenshots/ with these names, then remove this comment markers.
![Dashboard](docs/screenshots/dashboard.png)
![Live session](docs/screenshots/live_session.png)
-->

## Quick start
```
git clone https://github.com/SampratiSinha/fitness-pose-correction.git
cd fitness-pose-correction
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
cd src
python main.py status
```
Add photos to `data/` first (see [data/README.md](data/README.md)), then:

| Command (from `src/`) | What it does |
|---|---|
| `python main.py demo` | Live webcam skeleton demo |
| `python main.py build` | Build the reference database |
| `python main.py compare` | MediaPipe vs OpenPose (needs the model in `models/`, see [models/README.md](models/README.md)) |
| `python main.py evaluate` | Score correct vs incorrect photos |
| `python main.py live --exercise squat` | Live session; add `--target-reps 10 --sets 3` and `--record` |
| `python main.py analyze --video ../my_set.mp4 --exercise squat` | Analyse a recorded video |
| `python main.py dashboard` | Open the integrated dashboard |
| `python main.py all` | Build, compare and evaluate in order |

MediaPipe is pinned to 0.10.21 because newer releases removed the `mp.solutions` API used here.

## Project structure
```
src/        main.py, build_database_images.py, comparison_engine.py, guidance.py, live_session.py,
            video_analysis.py, evaluate_scoring.py, compare_hpe_methods.py, pose_extraction.py, app.py
database/   generated database, profiles and result summaries
docs/       literature survey, presentation, comparison spreadsheet
data/ test/ models/   not uploaded, see the README in each folder
```

## How scoring works
Joint angles (elbow, knee, hip, body line, torso lean) are compared with the mean angle of the matching phase. Each joint scores 0 to 1 using a tolerance (twice the standard deviation, at least 10 degrees). The overall score is half the average and half the worst joint. Reps are counted when the smoothed primary angle goes high, then low, then high again, using thresholds derived from the reference photos.

## Limitations
One person, side view, good lighting; small reference set (about 6 photos per exercise); angles are 2D; "incorrect" photos were chosen manually; MediaPipe and OpenPose agree less at the hips and knees than at other joints.

## Documentation
- [Literature survey (PDF)](docs/Literature_Survey.pdf)
- [Presentation](docs/HPE_Fitness_Progress_Review.pptx)
- [MediaPipe vs OpenPose results (spreadsheet)](docs/HPE_Comparison_MediaPipe_vs_OpenPose.xlsx)

## License
MIT, see [LICENSE](LICENSE).
