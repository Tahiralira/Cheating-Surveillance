import os
import cv2
import torch
import random
from facenet_pytorch import MTCNN
from imutils.video import WebcamVideoStream
import time
import datetime
import keyboard
import pygetwindow as gw
import threading
import pyperclip
import getpass
import sqlite3

app_data_dir = os.getenv('APPDATA')

# Define the directory name to store log files
log_dir = os.path.join(app_data_dir, 'KeystrokeLogger')

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logged_in_user = getpass.getuser()

conn = sqlite3.connect('surveillance.db')
cursor = conn.cursor()

# Read contents of text files
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

# Insert data into database
def insert_data_into_db(user, keystroke_log_file, window_tab_log_file, cam_log_file):
    keystroke_log = read_text_file(keystroke_log_file)
    window_tab_log = read_text_file(window_tab_log_file)
    cam_log = read_text_file(cam_log_file)
    cursor.execute('''INSERT INTO user_logs (user, keystroke_log, window_tab_log, cam_log) VALUES (?, ?, ?, ?)''',
                   (user, keystroke_log, window_tab_log, cam_log))
    conn.commit()
    
# Define the log file paths within the log directory
KEYSTROKE_LOG_FILE = os.path.join(
    log_dir, f"{logged_in_user}_keystroke_log.txt")
WINDOW_LOG_FILE = os.path.join(log_dir, f"{logged_in_user}_window_log.txt")
WEBCAM_LOG_FILE = os.path.join(
    log_dir, f"{logged_in_user}_webcam_log.txt")




def detect_faces(mtcnn, frame):
    # Detect faces and facial landmarks
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        # Calculate the vertical distance between the eyes and the nose
        left_eye_y = landmarks[0][0][1]
        right_eye_y = landmarks[0][1][1]
        nose_y = landmarks[0][2][1]
        eye_to_nose_distance = nose_y - (left_eye_y + right_eye_y) / 2
        # print("Eye to nose distance:", eye_to_nose_distance)  # For debugging
        # Check if the distance is within a reasonable range
        if 5 <= eye_to_nose_distance <= 20:  # Adjust this range as needed
            print("Looking at Screen")
            return "Looking at Screen"
        elif eye_to_nose_distance > 40:  # Adjust this threshold as needed
            print("Looking down, Possible Mobile Usage")
            return "Looking down/Mobile Usage"
        else:
            # Get the position of the first detected face
            x, _, w, _ = map(int, boxes[0])
            center = (x + w) / 2
            width = frame.shape[1]
            # Determine position based on the relative position of the face in the frame
            if center < width * 0.4:
                print("Looking at Right")
                return "Looking at Right"
            elif center > width * 0.6:
                print("Looking at Left")
                return "Looking at Left"
            else:
                print("Looking at Screen")
                return "Looking at Screen"

    else:
        print("No Face Detected or Possible Mobile Use")
        return "No Face/Using Mobile"


def save_frame_as_proof(frame, filename_prefix):
    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Construct the filename with the timestamp
    filename = f"{filename_prefix}_{timestamp}.jpg"
    # Save the grayscale frame as a JPEG image with compression quality 90
    # (according to my calculations it will be a 9 Mb file for a complete Section with at most 6 Pictures Per Student)
    cv2.imwrite(filename, grayscale_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return filename


def main():
    # Check if GPU is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize MTCNN
    mtcnn = MTCNN(device=device)

    try:
        with open(WEBCAM_LOG_FILE, "a") as log_file:
            while True:
                # Randomly wait between 1 and 5 minutes before capturing the next frame
                # Random time in seconds (1 to 5 minutes)
                wait_time = random.randint(1, 3)
                print(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)

                # Open webcam
                video_capture = WebcamVideoStream(src=0).start()
                time.sleep(2.0)  # Warm-up time for camera

                # Capture frame-by-frame
                frame = video_capture.read()

                # Use threading for face detection
                position = detect_faces(mtcnn, frame)

                # Draw text indicating the position of the head
                cv2.putText(
                    frame, f"Position: {position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save the frame as proof with a timestamp in the filename
                proof_filename = save_frame_as_proof(frame, "proof")

                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                # Write the position, timestamp, and proof filename to the log file
                with open(WEBCAM_LOG_FILE, "a") as log_file:
                    log_file.write(
                        f"Timestamp: {timestamp}, Position: {position}, Proof Frame: {proof_filename}\n")
# Display the resulting frame
                # cv2.imshow('Video', frame)
                # cv2.waitKey(0)  # Wait for any key press
                # Close the webcam window after 1 second
                time.sleep(1)
                cv2.destroyAllWindows()
                video_capture.stop()

    except KeyboardInterrupt:
        print("Exiting...")


def ensure_log_directory_exists():
    os.makedirs(os.path.dirname(KEYSTROKE_LOG_FILE), exist_ok=True)


def get_active_window_title():
    active_window = gw.getActiveWindow()
    if active_window is not None:
        return active_window.title
    else:
        return "Unknown"


def log_keystroke(keys, window_title):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ensure_log_directory_exists()
    with open(KEYSTROKE_LOG_FILE, 'a') as f:
        f.write(f"{timestamp}: {keys} - {window_title}\n")


def on_press(event):
    window_title = get_active_window_title()
    if keyboard.is_pressed('ctrl') and event.name == 'c':
        log_keystroke('Ctrl+C', window_title)
    elif keyboard.is_pressed('ctrl') and event.name == 'v':
        log_keystroke('Ctrl+V', window_title)
    elif keyboard.is_pressed('alt') and event.name == 'tab':
        log_keystroke('Alt+Tab', window_title)
    elif keyboard.is_pressed('ctrl') and event.name == 'a':
        log_keystroke('Ctrl+A', window_title)
    elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('win') and event.name == 'left':
        log_keystroke('Ctrl+Win+Left', window_title)
    elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('win') and event.name == 'right':
        log_keystroke('Ctrl+Win+Right', window_title)


def monitor_clipboard():
    previous_clipboard_content = None
    while True:
        clipboard_content = pyperclip.paste()
        if clipboard_content != previous_clipboard_content:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(KEYSTROKE_LOG_FILE, 'a') as f:
                f.write(f"{timestamp} - Clipboard: {clipboard_content}\n")
            previous_clipboard_content = clipboard_content
        time.sleep(1)


previous_clipboard_content = None


def check_clipboard():
    global previous_clipboard_content

    while True:
        clipboard_content = pyperclip.paste()
        if clipboard_content != previous_clipboard_content:
            # print("New clipboard content:", clipboard_content)
            previous_clipboard_content = clipboard_content
            # Log the clipboard content
            with open(WINDOW_LOG_FILE, "a") as log_file:
                log_file.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Clipboard: {clipboard_content}\n")

        time.sleep(1)  # Adjust the polling interval as needed


def log_active_window_change():
    previous_active_window = None

    while True:
        active_window = gw.getActiveWindow()
        if active_window != previous_active_window:
            if active_window is not None:
                # print(f"Switched to: {active_window.title}")
                # Log the window title
                with open(WINDOW_LOG_FILE, "a") as log_file:
                    log_file.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Switched to: {active_window.title}\n")
            previous_active_window = active_window
        time.sleep(1)  # Adjust the polling interval as needed


if __name__ == "__main__":
    threads = []
    threads.append(threading.Thread(target=main))
    threads.append(threading.Thread(target=keyboard.on_press, args=(on_press,)))
    threads.append(threading.Thread(target=log_active_window_change))
    threads.append(threading.Thread(target=monitor_clipboard))
    threads.append(threading.Thread(target=check_clipboard))

    # Set all threads as daemon threads so they will terminate when the main thread terminates
    for thread in threads:
        thread.daemon = True
        thread.start()

    try:
        while True:
            # Your main loop logic goes here
            time.sleep(1)

            
    except KeyboardInterrupt:
        insert_data_into_db(logged_in_user, KEYSTROKE_LOG_FILE, WINDOW_LOG_FILE, WEBCAM_LOG_FILE)
        conn.close()
        cursor.close()
        print("Exiting...")
