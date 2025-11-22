import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# --- Setup MediaPipe and Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# --- Configuration ---
# 1. POSE (Arm/Body Landmarks)
POSE_INDICES_TO_TRACK = [11, 12, 13, 14, 15, 16]  # Shoulders, Elbows, Wrists
POSE_LANDMARK_NAMES = {
    11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
    13: 'LEFT_ELBOW', 14: 'RIGHT_ELBOW',
    15: 'LEFT_WRIST_POSE', 16: 'RIGHT_WRIST_POSE',  # Differentiate from Hand Wrist
}

# 2. HANDS (Detailed Finger Landmark)
HANDS_INDICES_TO_TRACK = [0, 8]  # 0: Wrist, 8: Index Finger Tip
HANDS_LANDMARK_NAMES = {
    0: 'WRIST_HAND',
    8: 'INDEX_FINGER_TIP'
}

# Combine all landmark indices and names for centralized processing
ALL_LANDMARK_NAMES = {**POSE_LANDMARK_NAMES, **HANDS_LANDMARK_NAMES}

# List to store all collected data records
data_records = []
frame_count = 0


# --- Function to save data to CSV ---
def save_data(data_list, filename="handover_data_log4.csv"):
    """Converts the list of records to a DataFrame and saves it as a CSV file."""
    if not data_list:
        print("No data collected to save.")
        return

    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False)
    print(f"\n--- Data saved successfully! ---")
    print(f"Total records: {len(data_list)}")
    print(f"File saved as: {os.path.abspath(filename)}")
    print(f"Columns recorded: {list(df.columns)}")


# --- Main Capture and Logging Loop ---
def run_handover_recorder():
    global frame_count

    # Initialize the video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize both Pose and Hands models
    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as pose, \
            mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.6,
                           max_num_hands=2) as hands:

        start_time = time.time()

        print("--- Dual-Model Handover Recorder Started (Arm + Fingers) ---")
        print("Press 'q' to stop recording and save the data.")

        # --- Window Sizing Configuration (Updated for Full Screen) ---
        WINDOW_NAME = 'MediaPipe Handover Recorder (Dual Tracking)'

        # Configure the window to open in full-screen mode for maximum size
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # -----------------------------------

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Get image dimensions for displaying pixel coordinates
            image_height, image_width, _ = image.shape

            # Performance optimization: Convert BGR to RGB for processing
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Process with Pose Model (for Arm/Shoulder)
            pose_results = pose.process(image_rgb)

            # 2. Process with Hands Model (for Detailed Fingers)
            hands_results = hands.process(image_rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            current_timestamp = time.time()
            relative_time = current_timestamp - start_time
            frame_count += 1

            # --- Data Logging, Drawing, and Coordinate Display ---

            # 1. Log and Draw POSE Landmarks (Arm/Shoulders)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # Log and display only the specific arm points
                for lm_index in POSE_INDICES_TO_TRACK:
                    landmark = landmarks[lm_index]

                    if landmark.visibility > 0.5:
                        # Log data
                        record = {
                            'frame_id': frame_count,
                            'timestamp': relative_time,
                            'source': 'POSE',
                            'landmark_name': POSE_LANDMARK_NAMES.get(lm_index, str(lm_index)),
                            'x_norm': landmark.x,
                            'y_norm': landmark.y,
                            'z_norm': landmark.z,
                            'visibility': landmark.visibility
                        }
                        data_records.append(record)

                        # --- Display Coordinates ---
                        # Convert normalized coordinates to pixel coordinates for text placement
                        x_pixel = int(landmark.x * image_width)
                        y_pixel = int(landmark.y * image_height)

                        coord_text = f"X:{landmark.x:.2f} Y:{landmark.y:.2f} Z:{landmark.z:.2f}"

                        cv2.putText(image,
                                    coord_text,
                                    (x_pixel + 10, y_pixel),  # Offset text slightly from the point
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)  # Yellow text
                        # ---------------------------

                # Draw Pose landmarks (will show arm structure)
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
                )

            # 2. Log and Draw HANDS Landmarks (Detailed Fingers)
            if hands_results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    handedness = hands_results.multi_handedness[hand_index].classification[0].label

                    # Log and display only the specified finger points
                    for lm_index in HANDS_INDICES_TO_TRACK:
                        landmark = hand_landmarks.landmark[lm_index]

                        # Log data
                        record = {
                            'frame_id': frame_count,
                            'timestamp': relative_time,
                            'source': f'HAND_{hand_index}',  # Distinguish between two hands
                            'landmark_name': HANDS_LANDMARK_NAMES.get(lm_index, str(lm_index)),
                            'x_norm': landmark.x,
                            'y_norm': landmark.y,
                            'z_norm': landmark.z,
                            'handedness': handedness
                        }
                        data_records.append(record)

                        # --- Display Coordinates ---
                        # Convert normalized coordinates to pixel coordinates for text placement
                        x_pixel = int(landmark.x * image_width)
                        y_pixel = int(landmark.y * image_height)

                        coord_text = f"X:{landmark.x:.2f} Y:{landmark.y:.2f} Z:{landmark.z:.2f}"

                        cv2.putText(image,
                                    coord_text,
                                    (x_pixel + 10, y_pixel + 15),  # Offset text slightly
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)  # Cyan text
                        # ---------------------------

                    # Draw Hand landmarks (will show finger details)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Blue for Hands
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1)
                        # Cyan for Hand Connections
                    )

            # Display info text on the video feed
            hands_count = len(hands_results.multi_hand_landmarks) if hands_results.multi_hand_landmarks else 0
            pose_status = 'Yes' if pose_results.pose_landmarks else 'No'

            cv2.putText(image, f"Hands: {hands_count} | Pose: {pose_status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Time: {relative_time:.2f}s | Frame: {frame_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "Press 'q' to stop and save data",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA, )

            # Show the video feed
            cv2.imshow(WINDOW_NAME, image)  # Use the defined variable

            # Exit condition
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # --- Cleanup and Save ---
    cap.release()
    cv2.destroyAllWindows()
    save_data(data_records)


if __name__ == '__main__':
    run_handover_recorder()
