from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Lock
from sklearn.ensemble import IsolationForest
from collections import deque

# Initialize Flask app
app = Flask(__name__)

# Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables for camera and pose
camera = None
pose = None
camera_lock = Lock()
camera_initialized = False

# For form analysis
class FormAnalyzer:
    def __init__(self, window_size=10):
        self.isolation_forest = IsolationForest(contamination=0.2, random_state=42)
        self.angles_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.balance_history = deque(maxlen=window_size)
        self.is_trained = False
        self.last_rep_time = time.time()
        
    def analyze_form(self, current_angles, landmarks):
        """Analyze form using IsolationForest and additional metrics."""
        if not isinstance(current_angles, list):
            current_angles = [current_angles]
            
        current_time = time.time()
        rep_duration = current_time - self.last_rep_time
        
        # Calculate balance score using shoulder alignment
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        balance_score = abs(l_shoulder.y - r_shoulder.y)  # Lower is better
        
        self.angles_history.append(current_angles)
        self.time_history.append(rep_duration)
        self.balance_history.append(balance_score)
        
        feedback = []
        
        # Analyze rep speed
        if rep_duration < 1.0:
            feedback.append("Slow down")
        elif rep_duration > 3.0:
            feedback.append("Speed up")
            
        # Analyze balance
        if balance_score > 0.1:  # Threshold for shoulder misalignment
            feedback.append("Stay balanced")
            
        # Analyze form using IsolationForest
        if len(self.angles_history) >= 3:
            if not self.is_trained or len(self.angles_history) % 5 == 0:
                features = np.array(list(self.angles_history))
                self.isolation_forest.fit(features)
                self.is_trained = True
            
            prediction = self.isolation_forest.predict([current_angles])
            is_normal_rep = prediction[0] == 1
            
            if not is_normal_rep:
                feedback.append("Maintain consistent form")
        
        # Reset timer for next rep
        self.last_rep_time = current_time
        
        # Determine overall form quality
        form_correct = len(feedback) == 0
        
        # Generate feedback message
        if form_correct:
            feedback_msg = "Perfect form!"
        else:
            feedback_msg = " + ".join(feedback)
            
        return form_correct, feedback_msg

form_analyzer = FormAnalyzer()

# Exercise state with thread safety
class ExerciseState:
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.feedback = "Ready to start!"
        self.form_feedback = "Analyzing form..."
        self.debug_info = ""
        self.lock = Lock()
        self.last_update = time.time()
        self.last_feedback = ""  # Store last feedback to prevent duplicate messages
        self.rep_history = []  # Store history of rep quality

    def increment_counter(self):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_update > 0.5:
                self.counter += 1
                self.last_update = current_time
                self.feedback = f"Good rep! Count: {self.counter}"

    def reset(self):
        with self.lock:
            self.counter = 0
            self.stage = "down"
            self.feedback = "Ready to start!"
            self.form_feedback = "Analyzing form..."
            self.debug_info = ""
            self.last_feedback = ""
            self.rep_history = []
            self.last_update = time.time()

state = ExerciseState()

def initialize_camera():
    """Initialize the camera with proper settings."""
    global camera, camera_initialized
    if camera is not None:
        camera.release()
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        # Try alternative camera index
        camera = cv2.VideoCapture(1)
    
    if camera.isOpened():
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera_initialized = True
    return camera.isOpened()

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        if np.any(np.isnan([a, b, c])):
            return None
            
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        return angle if angle <= 180 else 360 - angle
    except:
        return None

def process_frame(frame, exercise_type):
    """Process a single frame for pose detection and exercise counting."""
    global pose
    
    if pose is None:
        pose = mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
    
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        feedback = "Start exercising!"
        form_correct = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract common landmarks
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            angles_to_analyze = []
            
            if exercise_type == "bicep_curl":
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                if angle is not None:
                    state.debug_info = f"Bicep Curl - Angle: {angle:.1f}°, Stage: {state.stage}"
                    angles_to_analyze.append(angle)
                    if angle > 130:  # Down position
                        if state.stage != "down":
                            state.stage = "down"
                            feedback = "Down position - Now curl up!"
                    elif angle < 70:  # Up position
                        if state.stage == "down":
                            state.stage = "up"
                            state.increment_counter()
                            
            elif exercise_type == "lateral_raise":
                angle = calculate_angle(l_hip, l_shoulder, l_elbow)
                if angle is not None:
                    state.debug_info = f"Lateral Raise - Angle: {angle:.1f}°, Stage: {state.stage}"
                    angles_to_analyze.append(angle)
                    if angle < 20:  # Down position
                        if state.stage != "down":
                            state.stage = "down"
                            feedback = "Down position - Raise arms!"
                    elif angle > 75:  # Up position
                        if state.stage == "down":
                            state.stage = "up"
                            state.increment_counter()
                            
            elif exercise_type == "shoulder_press":
                angle1 = calculate_angle(l_hip, l_shoulder, l_elbow)
                angle2 = calculate_angle(l_shoulder, l_elbow, l_wrist)
                if angle1 is not None and angle2 is not None:
                    state.debug_info = f"Shoulder Press - Angles: {angle1:.1f}°, {angle2:.1f}°, Stage: {state.stage}"
                    angles_to_analyze.extend([angle1, angle2])
                    if angle1 < 45 and angle2 > 150:  # Down position
                        if state.stage != "down":
                            state.stage = "down"
                            feedback = "Down position - Press up!"
                    elif angle1 > 80 and angle2 < 60:  # Up position
                        if state.stage == "down":
                            state.stage = "up"
                            state.increment_counter()

            # Analyze form using enhanced FormAnalyzer and update wireframe color
            if angles_to_analyze:
                form_correct, feedback_msg = form_analyzer.analyze_form(angles_to_analyze, landmarks)
                state.form_feedback = feedback_msg
                
                # Determine wireframe color based on form quality
                wireframe_color = (0, 255, 0) if form_correct else (0, 0, 255)  # Green for good form, Red for bad form

                # Display the feedback message prominently
                if state.stage == "up" and feedback_msg != state.last_feedback:
                    cv2.putText(image, feedback_msg,
                              (int(image.shape[1]/2) - 150, int(image.shape[0]/2)),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1.2, (0, 255, 255), 2)
                state.last_feedback = feedback_msg

                # Draw pose landmarks with dynamic color
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=wireframe_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=wireframe_color, thickness=2, circle_radius=2)
                )

                # Draw workout information
                cv2.putText(image, f"Reps: {state.counter}", 
                          (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.2, (0, 255, 0), 2)
                
                cv2.putText(image, f"Stage: {state.stage}", 
                          (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, (245, 117, 66), 2)
                
                # Draw form feedback in color based on correctness
                cv2.putText(image, state.form_feedback, 
                          (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, (0, 255, 0) if form_correct else (0, 0, 255), 2)
                
                # Draw debug info
                cv2.putText(image, state.debug_info, 
                          (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, (255, 255, 255), 2)

        return image
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame

def generate_frames(exercise_type):
    """Generate frames for video streaming."""
    global camera, camera_initialized
    
    if not camera_initialized:
        if not initialize_camera():
            print("Error: Could not initialize camera.")
            return
    
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                if not initialize_camera():
                    print("Error: Camera disconnected and could not be reinitialized.")
                    break
            
            success, frame = camera.read()
            
            if not success:
                print("Error: Failed to read frame from camera.")
                continue
            
            try:
                processed_frame = process_frame(frame, exercise_type)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                print(f"Error generating frame: {str(e)}")
                continue

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed/<exercise_type>')
def video_feed(exercise_type):
    """Video streaming route."""
    return Response(generate_frames(exercise_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    """Get the current rep count and feedback."""
    return jsonify({
        'count': state.counter,
        'feedback': state.feedback,
        'form_feedback': state.form_feedback,
        'debug_info': state.debug_info
    })

@app.route('/reset_count')
def reset_count():
    """Reset the exercise counter."""
    state.reset()
    return jsonify({'status': 'success'})

def cleanup():
    """Clean up resources."""
    global camera, pose
    if camera is not None:
        camera.release()
    if pose is not None:
        pose.close()

# Initialize camera when app starts
with app.app_context():
    initialize_camera()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
    finally:
        cleanup()