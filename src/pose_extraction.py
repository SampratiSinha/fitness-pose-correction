import time
import cv2
import mediapipe as mp

CAMERA_INDEX = 0
MODEL_VARIANT = "full"
FLIP_HORIZONTAL = True

_MODEL_COMPLEXITY = {"lite": 0, "full": 1, "heavy": 2}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_pose_from_image(image_bgr, model_complexity=1):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = [
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in results.pose_landmarks.landmark
        ]
        return landmarks


def run_webcam_demo(camera_index=CAMERA_INDEX, model_variant=MODEL_VARIANT):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}. "
              f"Try a different CAMERA_INDEX (e.g. 1) at the top of this file.")
        return

    model_complexity = _MODEL_COMPLEXITY.get(model_variant, 1)

    prev_time = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        print("Webcam started. Press 'q' in the video window to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera. Exiting.")
                break

            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("MediaPipe Pose - press 'q' to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()