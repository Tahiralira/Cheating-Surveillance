import cv2
import dlib
import numpy as np

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the angle between three points (two eyes and the nose)
def angle_between_points(a, b, c):
    ang = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.degrees(ang)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract coordinates of the left and right eye and the tip of the nose
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)

        # Calculate the angle between the eyes and the nose tip
        angle = angle_between_points(left_eye, right_eye, nose_tip)

        # Display the angle on the frame
        cv2.putText(frame, f'Angle: {angle:.2f} degrees', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if the angle exceeds 25 degrees to the left or right
        if abs(angle) > 65:
            cv2.putText(frame, 'Cheating Possible!', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Cheating")

    # Display the frame
    cv2.imshow('Head Movement Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
