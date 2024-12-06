import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.ensemble import IsolationForest

# Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate lateral raise angle
def calculate_angle_for_lateral_raise(shoulder, wrist):
    """
    Calculate the angle of the arm relative to the horizontal plane
    passing through the shoulder.
    """
    horizontal_reference = np.array([1, 0])  # Horizontal vector
    arm_vector = np.array([wrist[0] - shoulder[0], wrist[1] - shoulder[1]])
    dot_product = np.dot(horizontal_reference, arm_vector)
    magnitude_reference = np.linalg.norm(horizontal_reference)
    magnitude_arm = np.linalg.norm(arm_vector)
    if magnitude_arm == 0 or magnitude_reference == 0:
        return 0
    cos_angle = dot_product / (magnitude_reference * magnitude_arm)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


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


# Function to check if all required joints are visible
def are_key_joints_visible(landmarks, visibility_threshold=0.5):
    """
    Ensure that all required joints are visible based on their visibility scores.
    """
    required_joints = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ]
    for joint in required_joints:
        if landmarks[joint].visibility < visibility_threshold:
            return False
    return True


# Real-time feedback for single rep
def analyze_single_rep(rep, rep_data):
    """Provide actionable feedback for a single rep."""
    feedback = []

    # Calculate averages from previous reps
    avg_rom = np.mean([r["ROM"] for r in rep_data]) if rep_data else 0
    avg_tempo = np.mean([r["Tempo"] for r in rep_data]) if rep_data else 0

    # Dynamic tempo thresholds
    lower_tempo_threshold = 2.0  # Minimum grace threshold for faster tempo
    upper_tempo_threshold = 9.0  # Maximum grace threshold for slower tempo

    # Adjust thresholds after a few reps
    if len(rep_data) > 3:
        lower_tempo_threshold = max(2.0, avg_tempo * 0.7)
        upper_tempo_threshold = min(9.0, avg_tempo * 1.3)

    # Feedback for ROM
    if rep["ROM"] < 30:  # Minimum ROM threshold
        feedback.append("Lift arm higher")
    elif rep_data and rep["ROM"] < avg_rom * 0.8:
        feedback.append("Increase ROM")

    # Feedback for Tempo
    if rep["Tempo"] < lower_tempo_threshold:  # Tempo too fast
        feedback.append("Slow down")
    elif rep["Tempo"] > upper_tempo_threshold:  # Tempo too slow
        feedback.append("Speed up")

    return feedback


# Post-workout feedback function
def analyze_workout_with_isolation_forest(rep_data):
    if not rep_data:
        print("No reps completed.")
        return

    print("\n--- Post-Workout Summary ---")

    # Filter valid reps for recalculating thresholds
    valid_reps = [rep for rep in rep_data if rep["ROM"] > 20]  # Ignore very low ROM reps

    if not valid_reps:
        print("No valid reps to analyze.")
        return

    features = np.array([[rep["ROM"], rep["Tempo"]] for rep in valid_reps])

    avg_rom = np.mean(features[:, 0])
    avg_tempo = np.mean(features[:, 1])
    std_rom = np.std(features[:, 0])
    std_tempo = np.std(features[:, 1])

    # Adjusted bounds for anomalies
    rom_lower_bound = max(20, avg_rom - std_rom * 2)
    tempo_lower_bound = max(1.0, avg_tempo - std_tempo * 2)
    tempo_upper_bound = min(10.0, avg_tempo + std_tempo * 2)

    print(f"ROM Lower Bound: {rom_lower_bound}")
    print(f"Tempo Bounds: {tempo_lower_bound}-{tempo_upper_bound}")

    # Anomaly detection
    for i, rep in enumerate(valid_reps, 1):
        feedback = []
        if rep["ROM"] < rom_lower_bound:
            feedback.append("Low ROM")
        if rep["Tempo"] < tempo_lower_bound:
            feedback.append("Too Fast")
        elif rep["Tempo"] > tempo_upper_bound:
            feedback.append("Too Slow")

        if feedback:
            print(f"Rep {i}: Anomalous | Feedback: {', '.join(feedback[:1])}")

    # Use Isolation Forest for secondary anomaly detection
    model = IsolationForest(contamination=0.1, random_state=42)  # Reduced contamination
    predictions = model.fit_predict(features)

    for i, prediction in enumerate(predictions, 1):
        if prediction == -1:  # Outlier
            print(f"Rep {i}: Isolation Forest flagged this rep as anomalous.")


# Main workout tracking function
def main():
    cap = cv2.VideoCapture(0)
    counter = 0  # Rep counter
    stage = None  # Movement stage
    feedback = []  # Real-time feedback for the video feed
    rep_data = []  # Store metrics for each rep
    angles_during_rep = []  # Track angles during a single rep
    workout_start_time = None  # Timer start

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
                    draw_text_with_background(
                        image, "Ensure all joints are visible", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 255)
                    )
                    cv2.imshow("Lateral Raise Tracker", image)
                    continue

                # Extract key joints
                left_shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                left_wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]

                # Calculate angle for lateral raise
                angle = calculate_angle_for_lateral_raise(left_shoulder, left_wrist)

                # Track angles during a rep
                if stage == "up" or stage == "down":
                    angles_during_rep.append(angle)

                # Stage logic for counting reps
                if angle < 20 and stage != "down":
                    stage = "down"
                    if counter == 10:  # Stop on the down stage of the 10th rep
                        print("Workout complete! 10 reps reached.")
                        break

                    # Calculate ROM for the completed rep
                    if len(angles_during_rep) > 1:
                        rom = max(angles_during_rep) - min(angles_during_rep)
                    else:
                        rom = 0.0

                    tempo = elapsed_time
                    print(f"Rep {counter + 1}: ROM={rom:.2f}, Tempo={tempo:.2f}s")

                    # Record metrics for the rep
                    rep_data.append({
                        "ROM": rom,
                        "Tempo": tempo,
                    })

                    # Reset angles and timer for the next rep
                    angles_during_rep = []
                    workout_start_time = time.time()  # Reset timer

                if 70 <= angle <= 110 and stage == "down":
                    stage = "up"
                    counter += 1

                    # Analyze feedback
                    feedback = analyze_single_rep(rep_data[-1], rep_data)

                # Determine wireframe color
                wireframe_color = (0, 255, 0) if not feedback else (0, 0, 255)

                # Display feedback
                draw_text_with_background(image, f"Reps: {counter}", (50, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, (0, 0, 0))
                draw_text_with_background(image, " | ".join(feedback), (50, 120),
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
            cv2.imshow("Lateral Raise Tracker", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Post-workout analysis
    analyze_workout_with_isolation_forest(rep_data)


if __name__ == "__main__":
    main()
