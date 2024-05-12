import cv2
from facenet_pytorch import MTCNN
import numpy as np

# Initialize MTCNN
mtcnn = MTCNN()

cap = cv2.VideoCapture(0) # Opens webcam (will hide it later)

# Thresholds for head movement (in pixels) and pupil movement (in distance change)
# Adjusted threshold values
HEAD_MOVEMENT_THRESHOLD = 10  # Increase for better sensitivity
PUPIL_MOVEMENT_THRESHOLD = 10  # Increase for better sensitivity
PUPIL_ANGLE_THRESHOLD = 5  # Decrease for better sensitivity  # Threshold for angle between eye vectors

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MTCNN expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and landmarks
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    if boxes is not None:
        for box, landmark in zip(boxes, landmarks):
            # Draw bounding box around face
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            # Adjust the bounding box size
            #expansion_factor = 1.5  # Increase this factor to enlarge the bounding box
            #expanded_box = [
           #     int(box[0] - (box[2] - box[0]) * (expansion_factor - 1) / 2),  # Adjust left coordinate
             #   int(box[1] - (box[3] - box[1]) * (expansion_factor - 1) / 2),  # Adjust top coordinate
             #   int(box[2] + (box[2] - box[0]) * (expansion_factor - 1) / 2),  # Adjust right coordinate
             #   int(box[3] + (box[3] - box[1]) * (expansion_factor - 1) / 2)   # Adjust bottom coordinate
           # ]

            # Draw expanded bounding box around face
            #cv2.rectangle(frame, (expanded_box[0], expanded_box[1]), (expanded_box[2], expanded_box[3]), (255, 0, 0), 2)

            # Draw landmarks
            for point in landmark:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            # Debugging output for landmark count
            print(f"Landmark count: {len(landmark)}")
            # Calculate movement in head and pupil
            if len(landmark) > 30 and len(landmark) > 4:
                head_movement = abs(landmark[30][1] - landmark[8][1])  # Vertical distance between nose and chin

                # Check for head movement alert
                if head_movement > HEAD_MOVEMENT_THRESHOLD:
                    print("Cheating")
                    cv2.putText(frame, "ALERT: Head movement detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check for pupil movement alert
            if len(landmark) > 42 and len(landmark) > 45:
                pupil_movement = np.linalg.norm(landmark[42] - landmark[45])  # Distance between left and right pupils

                if pupil_movement > PUPIL_MOVEMENT_THRESHOLD:
                    cv2.putText(frame, "ALERT: Pupil movement detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Calculate angle between eye vectors
            # Calculate angle between eye vectors
            if len(landmark) >= 48 and len(landmark) >= 42:  # Ensure all necessary landmarks for eyes are detected
                left_eye_center = np.mean(landmark[36:42], axis=0)
                right_eye_center = np.mean(landmark[42:48], axis=0)

                angle = calculate_angle(landmark[39], left_eye_center, right_eye_center)

                # Print values for debugging
                print(f"Angle: {angle}")

                # Check for pupil corner movement alert
                if angle > PUPIL_ANGLE_THRESHOLD:
                    cv2.putText(frame, "ALERT: Pupil corner movement detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Face and Landmark Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
