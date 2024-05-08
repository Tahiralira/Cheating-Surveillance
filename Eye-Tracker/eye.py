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

# blinking ka threshold
EAR_THRESHOLD = 0.2

cap = cv2.VideoCapture(0)#opens webcam need to hide the webcam later

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
        #this draws the boxes around the face and eye
        cv2.drawContours(frame, [np.array(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [np.array(right_eye)], -1, (0, 255, 0), 1)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        
        if avg_ear < EAR_THRESHOLD:
            cv2.putText(frame, "ALERT: Eyes looking elsewhere!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
