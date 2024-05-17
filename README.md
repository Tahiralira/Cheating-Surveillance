# Cheating-Surveillance
A Programming Lab Cheating Surveillance Software That Uses Keystroke/Mouse/Webcam Sequences To Detect Possible Cheating Scenarios In a Lab Exam Setting Using MT-Cascaded Nueral Networks, Reinforcement Learning Agents and Deep Q-Networks

Reminder = Requires External Download of CMake in the System


## Cheating Surveillance System

### Project Overview
This Cheating Surveillance System is designed to monitor user activities on a computer to detect potential cheating or unethical behavior. It logs keystrokes, mouse movements, clipboard contents, and captures frames from a webcam to analyze the user's focus and actions. The system includes a Flask backend that processes and serves the logged data, and a frontend that displays the data and allows interaction, such as viewing captured images and analyzing risk levels associated with the logs.

### Key Features
- **Keystroke Logging:** Captures all keystrokes along with timestamps and the active window titles.
- **Mouse Movement Logging:** Tracks mouse movements and logs window switches and other significant events.
- **Clipboard Monitoring:** Records any text that is copied to the clipboard.
- **Webcam Surveillance:** Captures frames based on specific triggers such as significant eye movement or leaving the workstation.
- **Risk Analysis:** Analyzes the collected data to assess the risk level of cheating or unethical behavior.
- **Dynamic Reporting:** Provides an interface to view detailed logs and the risk analysis results.
- **Feedback System:** Allows users to provide feedback on the system's risk assessment, which is used to train a reinforcement learning model to improve accuracy.

### Technologies Used
- **Python:** Core backend development.
- **Flask:** Server-side web framework used for handling web requests and serving the web application.
- **HTML/CSS/JavaScript:** Frontend development for displaying data and interacting with the backend.
- **Bootstrap:** For responsive design and styled components.
- **Reinforcement Learning:** Used to enhance risk analysis based on user feedback.

### Project Structure
- `app.py`: Main Flask application file with route definitions.
- `ScoringModel.py`: Contains the logic for parsing logs, calculating cheating scores, and managing the reinforcement learning agent.
- `templates/`: Contains HTML files for the web interface.
- `static/`: Contains CSS, JavaScript, and other static files.
- `Eye-Tracker/`: Directory storing captured frames and webcam logs.
- `Keylogger/`: Directory containing keystroke logs.
- `mousemovement/`: Stores logs related to mouse movements.

### Setup and Running the Project
1. **Clone the Repository:**
   ```
   git clone https://github.com/Tahiralira/Cheating-Surveillance.git
   cd Cheating-Surveillance
   ```

2. **Install Dependencies:**
   To install the required dependencies, run the following command:

   ```bash
   pip install -r requirements.txt
   ```

This command will install all the necessary packages listed in the `requirements.txt` file.

3. **Start the Flask Application:**
   ```
   python app.py
   ```

4. **Access the Web Interface:**
   - Open a web browser and navigate to `http://127.0.0.1:5000/` to view the interface.

5. ## Build Requirements

This project uses CMake as the build system. Before building the project, make sure you have CMake installed on your system.

### Installing CMake

You can download and install CMake from the [official website](https://cmake.org/download/). Alternatively, you can use your system's package manager to install CMake.

#### Example (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install cmake
```

### Usage

- **View Logs:** Click on the respective buttons to load different types of logs (keyboard, mouse, webcam).
- **Analyze Risk:** Click on the 'Analyze Logs' button to see the risk analysis based on the collected data.
- **Submit Feedback:** Use the feedback form to submit your assessment of the risk level, which will help train the system.

### Contributing
Contributions to the project are welcome. Please follow the standard pull request process to submit enhancements or fixes.

### License
This project is licensed under me.

---

### Author
Aheed Tahir
