import cv2
import dlib
from scipy.spatial import distance
import numpy as np

# face and eye box using external library dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_eyes(frame, face):
    landmarks = predictor(frame, face)
    left_eye = []
    right_eye = []
    for n in range(36, 42):  # Left eye 
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        left_eye.append((x, y))
    for n in range(42, 48):  # Right eye 
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        right_eye.append((x, y))
    return left_eye, right_eye


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_pupil(eye_region):
    # Convert eye region to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    # Thresholding to binarize the image
    _, threshold = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If contours are found
    if contours:
        # Get the largest contour (pupil)
        pupil_contour = max(contours, key=cv2.contourArea)
        # Calculate centroid of the pupil contour
        M = cv2.moments(pupil_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

# blinking ka threshold
EAR_THRESHOLD = 0.2

cap = cv2.VideoCapture(0) # Opens webcam (will hide it later)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        left_eye, right_eye = detect_eyes(frame, face)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        cv2.drawContours(frame, [np.array(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [np.array(right_eye)], -1, (0, 255, 0), 1)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        #if avg_ear < EAR_THRESHOLD:
        #    cv2.putText(frame, "ALERT: Eyes looking elsewhere!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Detect pupil for each eye
        for eye_landmarks in [left_eye, right_eye]:
            eye_region = frame[min([y for (x, y) in eye_landmarks]):max([y for (x, y) in eye_landmarks]),
                               min([x for (x, y) in eye_landmarks]):max([x for (x, y) in eye_landmarks])]
            pupil_coords = detect_pupil(eye_region)
            if pupil_coords:
                cx, cy = pupil_coords
                # Get the center of the eye region
                eye_center = ((eye_landmarks[0][0] + eye_landmarks[3][0]) // 2, (eye_landmarks[0][1] + eye_landmarks[3][1]) // 2)
                # Calculate distance between eye center and pupil
                dist = distance.euclidean(eye_center, pupil_coords)
                # If pupil is significantly away from the eye center, trigger alert
                if dist > 10:  # You may need to adjust this threshold based on your setup
                    cv2.putText(frame, "ALERT: Pupil looking away!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (cx + eye_landmarks[0][0], cy + eye_landmarks[0][1]), 2, (0, 0, 255), -1)

    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
