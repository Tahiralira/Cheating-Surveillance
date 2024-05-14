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
from collections import deque
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

FEEDBACK_FILE = 'feedback.json'
DETAILED_FEEDBACK_FILE = 'detailed_feedback.json'

def load_feedback(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []
dqn_agent_analysis = {}

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy()))
            target_f = self.model(torch.FloatTensor(state))
            target_f = target_f.detach().numpy()
            target_f[action] = target
            self.model.zero_grad()
            outputs = self.model(torch.FloatTensor(state))
            loss = self.criterion(outputs, torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

app_data_dir = os.getenv('APPDATA')

# Define the directory name to store log files
log_dir = os.path.join(app_data_dir, 'KeystrokeLogger')

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logged_in_user = getpass.getuser()

conn = sqlite3.connect('surveillance.db')
cursor = conn.cursor()
DATABASE = 'surveillance.db'

def read_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    
def insert_data_into_db(user, keystroke_log_file, window_tab_log_file, cam_log_file):
    keystroke_log = read_text_file(keystroke_log_file)
    window_tab_log = read_text_file(window_tab_log_file )
    cam_log = read_text_file(cam_log_file)

    if keystroke_log is None or window_tab_log is None or cam_log is None:
        print("One or more log files are missing. Skipping database insertion.")
        return    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            # Insert data into the database
            cursor.execute('''
                INSERT INTO user_logs (user, keystroke_log, window_log, cam_log)
                VALUES (?, ?, ?, ?)
            ''', (user, keystroke_log, window_tab_log, cam_log))
            conn.commit()
            print("Data inserted successfully")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
# Define the log file paths within the log directory
KEYSTROKE_LOG_FILE = os.path.join(
    log_dir, f"{logged_in_user}_keystroke_log.txt")
WINDOW_LOG_FILE = os.path.join(log_dir, f"{logged_in_user}_window_log.txt")
WEBCAM_LOG_FILE = os.path.join(
    log_dir, f"{logged_in_user}_webcam_log.txt")


import shutil
TARGET_LOG_FILE = os.path.join("Eye-Tracker", "webcam_log.txt")

def copy_webcam_log():
        #time.sleep(60)
        try:
            # Copy the contents of WEBCAM_LOG_FILE to TARGET_LOG_FILE
            shutil.copyfile(WEBCAM_LOG_FILE, TARGET_LOG_FILE)
            print("Webcam log updated successfully.")
        except Exception as e:
            print("Error:", e)
             

has_run = False

def run_once():
    global has_run
    if not has_run:
        print("Running function for the first time.")
        copy_webcam_log()
        has_run = True
    else:
        print("Function has already run.")





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
    # Define the directory where you want to save the image, using a forward slash as Flask and web use forward slashes
    directory = "Eye-Tracker"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Use os.path.join for file system operations but replace backslashes for web compatibility
    filepath = os.path.join(directory, filename).replace('\\', '/')
    # Save the grayscale frame as a JPEG image with compression quality 90
    cv2.imwrite(filepath, grayscale_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return filepath



def main():
    # Check if GPU is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MTCNN
    mtcnn = MTCNN(device=device)
    state_size = 1  # Assuming 1-dimensional state for simplicity
    action_size = 4  # Number of possible actions: [Look Screen, Look Right, Look Left, Look Down]
    agent = DQNAgent(state_size, action_size)

    # Load feedback from JSON files
    feedback_data = load_feedback(FEEDBACK_FILE)
    detailed_feedback_data = load_feedback(DETAILED_FEEDBACK_FILE)

    # Process feedback data
    for feedback in feedback_data:
        frame_index = feedback.get('frame_index')
        if frame_index is None:
            print("Invalid feedback data: 'frame_index' is missing or None.")
            continue
        state = [int(frame_index)]  # Simplified for demonstration
        action = 0 if feedback['feedback'] == 'yes' else 1
        reward = 1 if feedback['feedback'] == 'yes' else -1
        next_state = state  # Simplified for demonstration
        done = True  # Since it's feedback, consider it as a terminal state
        agent.memorize(state, action, reward, next_state, done)

    for detailed_feedback in detailed_feedback_data:
        state = [int(frame_index)]  # Simplified for demonstration
        action_mapping = {
            "looking_at_screen": 0,
            "looking_at_right": 1,
            "looking_at_left": 2,
            "looking_down_mobile": 3,
            "no_face_using_mobile": 4
        }
        face_position = detailed_feedback.get('face_position')
        if face_position not in action_mapping:
            print(f"Invalid face position: {face_position}")
            continue
        action = action_mapping[face_position]
        reward = 1 if face_position == "looking_at_screen" else -1
        next_state = state  # Simplified for demonstration
        done = True  # Since it's feedback, consider it as a terminal state
        agent.memorize(state, action, reward, next_state, done)

    try:
        with open("webcam_log.txt", "a") as log_file:
            while True:
                # Randomly wait between 1 and 5 minutes before capturing the next frame
                wait_time = random.randint(1, 3)  # Random time in seconds (1 to 5 minutes)
                print(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                
                # Open webcam
                video_capture = WebcamVideoStream(src=0).start()
                time.sleep(2.0)  # Warm-up time for camera
                
                # Capture frame-by-frame
                frame = video_capture.read()
                
                # Use threading for face detection
                position = detect_faces(mtcnn, frame)
                
                if position == "Looking at Screen":
                    state = [0]
                elif position == "Looking at Right":
                    state = [1]
                elif position == "Looking at Left":
                    state = [2]
                elif position == "Looking Down/Mobile Possible":
                    state = [3]
                else:
                    state = [4]   #No Face/Using Mobile
                
                action = agent.act(state)
                reward = 1 if position == "Looking at Screen" else -1
                next_state = state  # Simplified for demonstration
                
                agent.memorize(state, action, reward, next_state, False)
                if len(agent.memory) > 32:
                    agent.replay(32)
                
                dqn_agent_analysis['Looking at Screen'] = agent.model(torch.FloatTensor([0])).detach().numpy().tolist()
                dqn_agent_analysis['Looking at Right'] = agent.model(torch.FloatTensor([1])).detach().numpy().tolist()
                dqn_agent_analysis['Looking at Left'] = agent.model(torch.FloatTensor([2])).detach().numpy().tolist()
                dqn_agent_analysis['Looking Down/Mobile Possible'] = agent.model(torch.FloatTensor([3])).detach().numpy().tolist()
                dqn_agent_analysis['No Face/Using Mobile'] = agent.model(torch.FloatTensor([4])).detach().numpy().tolist()

                with open("dqn_agent_analysis.json", "w") as dqn_analysis_file:
                    json.dump(dqn_agent_analysis, dqn_analysis_file)
                
                # Draw text indicating the position of the head
                cv2.putText(frame, f"Position: {position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save the frame as proof with a timestamp in the filename
                proof_filename = save_frame_as_proof(frame, "proof")
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                
                # Write the position, timestamp, and proof filename to the log file
                with open(WEBCAM_LOG_FILE, "a") as log_file:
                    log_file.write(f"Timestamp: {timestamp}, Position: {position}, Proof Frame: {proof_filename}\n")
                    
                    # Display the resulting frame
                # cv2.imshow('Video', frame)
                # cv2.waitKey(0)  # Wait for any key press
                # Close the webcam window after 1 second
                time.sleep(1)
                cv2.destroyAllWindows()
                video_capture.stop()
                       
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Ensure resources are released and data is saved before exiting
        insert_data_into_db('user', 'keystroke_log.txt', 'window_tab_log.txt', 'cam_log.txt')

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
    print(f"Feedback file path: {FEEDBACK_FILE}")
    print(f"Detailed feedback file path: {DETAILED_FEEDBACK_FILE}")
    print(f"Database file path: {DATABASE}")
    print(f"Log directory path: {log_dir}")
    print(f"Keystroke log file path: {KEYSTROKE_LOG_FILE}")
    print(f"Window log file path: {WINDOW_LOG_FILE}")
    print(f"Webcam log file path: {WEBCAM_LOG_FILE}")
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
