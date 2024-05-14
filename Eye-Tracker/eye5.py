import cv2
import torch
import random
from facenet_pytorch import MTCNN
from imutils.video import WebcamVideoStream
import time
import datetime
import os
from collections import deque
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
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

def detect_faces(mtcnn, frame):
    # Detect faces and facial landmarks
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        # Calculate the vertical distance between the eyes and the nose
        left_eye_y = landmarks[0][0][1]
        right_eye_y = landmarks[0][1][1]
        nose_y = landmarks[0][2][1]
        eye_to_nose_distance = nose_y - (left_eye_y + right_eye_y) / 2
        #print("Eye to nose distance:", eye_to_nose_distance)  # For debugging
        # Check if the distance is within a reasonable range
        if 5 <= eye_to_nose_distance <= 20:  # Adjust this range as needed
            print("Looking at Screen")
            return "Looking at Screen"
        elif eye_to_nose_distance > 40:  # Adjust this threshold as needed
            print("Looking Down, Possible Mobile Usage")
            return "Looking Down/Mobile Possible"
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
    # Define the directory where you want to save the image
    directory = "Eye-Tracker"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
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
                proof_filename=save_frame_as_proof(frame, "proof")
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                
                # Write the position, timestamp, and proof filename to the log file
                log_file.write(f"Timestamp: {timestamp}, Position: {position}, Proof Frame: {proof_filename}\n")
                # Display the resulting frame
                #cv2.imshow('Video', frame)
                #cv2.waitKey(0)  # Wait for any key press
                # Close the webcam window after 1 second
                time.sleep(1)
                cv2.destroyAllWindows()
                video_capture.stop()
                
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
