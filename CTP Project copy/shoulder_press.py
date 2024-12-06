import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angles
def calculate_angle(point_a, point_b, point_c):
    vector_ab = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])
    vector_cb = np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])
    dot_product = np.dot(vector_ab, vector_cb)
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_cb = np.linalg.norm(vector_cb)
    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0
    cos_angle = dot_product / (magnitude_ab * magnitude_cb)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


# Function to check if all required joints are visible
def are_key_joints_visible(landmarks, visibility_threshold=0.5):
    required_joints = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ]
    for joint in required_joints:
        if landmarks[joint].visibility < visibility_threshold:
            return False
    return True


# Function to draw text with a background
def draw_text_with_background(image, text, position, font, font_scale, color, thickness, bg_color, padding=10):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    box_coords = (
        (text_x - padding, text_y - padding),
        (text_x + text_size[0] + padding, text_y + text_size[1] + padding),
    )
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y + text_size[1]), font, font_scale, color, thickness)


# Main workout tracking function
def main():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    feedback = ""
    workout_start_time = None
    rep_start_time = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Initialize workout start time
            if workout_start_time is None:
                workout_start_time = time.time()

            # Timer
            elapsed_time = time.time() - workout_start_time
            timer_text = f"Timer: {int(elapsed_time)}s"

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if pose landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Check if key joints are visible
                if not are_key_joints_visible(landmarks):
                    feedback = "Ensure all joints are visible"
                    draw_text_with_background(
                        image, feedback, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 255)
                    )
                    cv2.imshow("Shoulder Press Tracker", image)
                    continue

                # Extract key joints for both arms
                left_shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                left_elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                left_wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]

                right_shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                ]
                right_elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                ]
                right_wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                ]

                # Calculate angles
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Check starting and ending positions
                if 80 <= left_elbow_angle <= 100 and 80 <= right_elbow_angle <= 100 and stage != "down":
                    stage = "down"
                    if counter == 10:
                        feedback = "Workout complete! 10 reps done."
                        draw_text_with_background(image, feedback, (50, 120),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 255))
                        cv2.imshow("Shoulder Press Tracker", image)
                        break
                    if rep_start_time is not None:
                        tempo = time.time() - rep_start_time
                        feedback = f"Rep {counter} completed! Tempo: {tempo:.2f}s"
                        rep_start_time = None
                elif left_elbow_angle > 160 and right_elbow_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1
                    rep_start_time = time.time()

                # Wireframe color
                wireframe_color = (0, 255, 0) if "completed" in feedback or "Good" in feedback else (0, 0, 255)

                # Display feedback
                draw_text_with_background(image, f"Reps: {counter}", (50, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 0))
                draw_text_with_background(image, feedback, (50, 120),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 0))
                draw_text_with_background(image, timer_text, (50, 190),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 0))

                # Render detections with wireframe color
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=wireframe_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=wireframe_color, thickness=2, circle_radius=2),
                )

            # Display the image
            cv2.imshow("Shoulder Press Tracker", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
