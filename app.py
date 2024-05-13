from flask import Flask, jsonify, send_from_directory, render_template,request,session
import os
from flask_cors import CORS
import ScoringModel
import numpy as np
from ScoringModel import CheatingRiskAgent
agent = CheatingRiskAgent()
app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'
@app.route('/keylogs')
def send_key_logs():
    key_log_path = "./Keylogger/keylogger.txt"
    logs = send_log_contents(key_log_path)
    print("Keylogs: ", logs)  # Debug output
    return logs

@app.route('/mouselogs')
def send_mouse_logs():
    mouse_log_path = "./mousemovement/windowtablogger.txt"
    logs = send_log_contents(mouse_log_path)
    print("Mouselogs: ", logs)  # Debug output
    return logs

@app.route('/webcamlogs')
def send_webcam_logs():
    return send_log_contents("./Eye-Tracker/webcam_log.txt")

def send_log_contents(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            logs = file.read()
        return jsonify({"data": logs})
    else:
        return jsonify({"data": "No logs available."})

@app.route('/score')
def score_page():
    try:
        logs, score = ScoringModel.process_logs()
        session['score'] = score  # Store score in session
        return render_template('score.html', score=score, logs=logs)
    except Exception as e:
        # Handle the exception, maybe log it or inform the user
        return f"Error generating score page: {str(e)}"

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    try:
        score = int(request.form['score'])
        user_risk_level = int(request.form['risk_level'])
        action_taken = agent.get_risk_level(score)
        print(f"Received feedback with score {score}, user risk level {user_risk_level}")
        agent.feedback(score, action_taken, user_risk_level)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/get-analysis')
def get_analysis():
    try:
        analysis = agent.get_latest_analysis()
        print(f"Sending analysis: {analysis}")
        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/images/<path:filename>')
def send_image(filename):
    return send_from_directory('Eye-Tracker', filename)

@app.route('/api/log-data')
def log_data():
    """Parse the webcam_log.txt file and return structured log data."""
    log_path = os.path.join('Eye-Tracker', 'webcam_log.txt')
    logs = []
    try:
        with open(log_path, 'r') as file:
            for line in file:
                if "Proof Frame:" in line:
                    parts = line.strip().split(', ')
                    timestamp = parts[0].split(': ')[1]
                    position = parts[1].split(': ')[1]
                    image_file = parts[2].split(': ')[1]
                    logs.append({'timestamp': timestamp, 'position': position, 'image_file': image_file})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': True, 'logs': logs})
#@app.route('/api/get-score')
#def get_score():
 #   main()
  #  logs, score = process_logs()  # Assume this function returns the logs and calculated score
   # return jsonify({'score': score, 'logs': logs})




@app.route('/')
def index():
    return render_template('index.html')


    

if __name__ == "__main__":
    app.run(debug=True)

