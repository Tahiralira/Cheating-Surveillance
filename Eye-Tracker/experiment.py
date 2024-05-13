import cv2
import torch
import random
from facenet_pytorch import MTCNN
from imutils.video import WebcamVideoStream
import time
import datetime
import pickle
from torchvision import transforms
import json
import torch.nn as nn
import torch.nn.functional as F

class HeadPoseNet(nn.Module):
    def __init__(self):
        super(HeadPoseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 3)  # Assuming output dimensions for yaw, pitch, roll

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = HeadPoseNet()

# Load the state dictionary
state_dict = torch.load('hopenet_lite_6MB.pkl', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode
print(type(state_dict))

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
    # (according to my calculations it will be a 9 Mb file for a complete Section with at most 60 Pictures Per Student)
    cv2.imwrite(filename, grayscale_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return filename

def preprocess(frame):
    size = (224, 224)  # Common input size for CNNs

    # Resize the image
    frame_resized = cv2.resize(frame, size)

    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to a PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for pre-trained on ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations
    input_tensor = transform(frame_rgb)

    # Add a batch dimension since PyTorch treats all inputs as batches
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor

def get_head_pose(frame, model):
    
    input_tensor = preprocess(frame)  # Define this function based on model requirements

    # Perform prediction
    with torch.no_grad():
        yaw, pitch, roll = model(input_tensor)  # Adjust based on actual model output

    return yaw, pitch, roll

def save_output(data, file_path="output.json"):
    with open(file_path, "w") as f:
        json.dump(data, f)    

def main():
    # Check if GPU is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MTCNN
    mtcnn = MTCNN(device=device)
    results = []  # List to store results
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
                input_tensor = preprocess(frame)
                
                # Model prediction
                with torch.no_grad():
                    output = model(input_tensor)
                
                results.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "output": output.tolist()  # Assuming output is tensor, convert to list
                })
                
                
                if len(results) >= 3:  # condition
                    save_output(results)
                    results = []
                
                # Use threading for face detection
                position = detect_faces(mtcnn, frame)
                
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
