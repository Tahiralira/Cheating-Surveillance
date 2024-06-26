# @app.route('/images/<path:filename>')
# def send_image(filename):
#     return send_from_directory('Eye-Tracker', filename)


from flask import Flask, jsonify, send_from_directory, render_template, request, session, url_for, redirect, send_file
import os
from flask_cors import CORS
import ScoringModel
import numpy as np
from ScoringModel import CheatingRiskAgent
import sqlite3
from flask import g
import json
import main
from main import DQNAgent
import webbrowser
from threading import Timer

state_size = 1
action_size = 4
dqnagent = DQNAgent(state_size, action_size)

FEEDBACK_FILE = 'feedback.json'
DETAILED_FEEDBACK_FILE = 'detailed_feedback.json'

agent = CheatingRiskAgent()
app = Flask(__name__)
CORS(app)

app.secret_key = 'your_secret_key'

DATABASE = 'surveillance.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.route('/api/log-data')
def log_data():
    image_directory = os.path.join(app.root_path, 'Eye-Tracker')
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
    image_files.sort(reverse=True)  # Assuming you want the most recent files first
    log_data = [{'image_file': file, 'timestamp': file.split('_')[1].split('.')[0]} for file in image_files]
    return jsonify({'success': True, 'logs': log_data})


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/homepage')
def homepage():
    cursor = get_db().cursor()
    cursor.execute('SELECT user FROM user_logs')
    users = cursor.fetchall()
    return render_template('index.html', users=users)

@app.route('/log_viewer/<user>')
def log_viewer(user):
    return render_template('log_viewer.html', user=user)

@app.route('/keylogs')
def get_key_logs():
    try:
        filepath = os.path.join(os.getenv('APPDATA'), 'KeystrokeLogger', 'user_keystroke_log.txt')
        with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
            logs = file.read()
        return jsonify({'data': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mouselogs')
def get_mouse_logs():
    try:
        filepath = os.path.join(os.getenv('APPDATA'), 'KeystrokeLogger', 'user_window_log.txt')
        with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
            logs = file.read()
        return jsonify({'data': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camlogs')
def get_cam_logs():
    try:
        filepath = os.path.join(os.getenv('APPDATA'), 'KeystrokeLogger', 'user_webcam_log.txt')
        with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
            logs = file.read()
        return jsonify({'data': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/score/<username>')
def score_page(username):
    try:
        # Process logs for the specified username
        logs, score = ScoringModel.process_logs(username)
        session['score'] = score  # Store score in session
        return render_template('score.html', username=username, score=score, logs=logs)
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

@app.route('/get-dqn-analysis', methods=['GET'])
def get_dqn_analysis():
    try:
        with open("dqn_agent_analysis.json", "r") as analysis_file:
            analysis_data = json.load(analysis_file)
        analysis_string = ", ".join([f"{k}: {v}" for k, v in analysis_data.items()])
        return jsonify({"success": True, "analysis": analysis_string})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if the username and password match a record in the database
        cursor = get_db().cursor()
        cursor.execute('''SELECT * FROM users WHERE username=? AND password=?''', (username, password))
        user = cursor.fetchone()
        if user:
            # If user exists, set session and redirect to authenticated route
            session['username'] = username
            return redirect(url_for('homepage'))
        else:
            # If authentication fails, redirect back to login page with error message
            return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html')

@app.route('/images/<path:filename>')
def send_image(filename):
    return send_from_directory('Eye-Tracker', filename)




@app.route('/logs/<username>')
def get_user_logs(username):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT log_content, image_filename FROM user_logs WHERE username=?", (username,))
        logs = [{'log': row[0], 'image': row[1]} for row in cursor.fetchall()]
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/parse-webcam-logs')
def parse_webcam_logs():
    # Run the background task
    main.run_once() 
    log_path = os.path.join(app.root_path, 'Eye-Tracker', 'webcam_log.txt')
    image_files = []
    try:
        with open(log_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "Proof Frame:" in line:
                    parts = line.split('Proof Frame: ')
                    if len(parts) > 1:
                        filename_with_path = parts[1].strip()
                        filename = os.path.basename(filename_with_path)  # Extract filename only
                        image_files.append(filename)
        return jsonify({'success': True, 'images': image_files})
    except FileNotFoundError:
        return jsonify({'success': False, 'error': 'Log file not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dqn_feedback', methods=['POST'])
def dqn_feedback():
    try:
        data = request.get_json()
        feedback = {
            'feedback': data.get('feedback'),
            'frame_index': data.get('frame_index')
        }
        save_feedback(feedback, FEEDBACK_FILE)
        update_dqn_analysis()  # Update DQN analysis after feedback submission
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dqn_detailed_feedback', methods=['POST'])
def dqn_detailed_feedback():
    try:
        feedback = {
            'face_position': request.form.get('face_position')
        }
        save_feedback(feedback, DETAILED_FEEDBACK_FILE)
        update_dqn_analysis()  # Update DQN analysis after detailed feedback submission
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def save_feedback(data, filename):
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
    with open(filename, 'r+') as f:
        feedback_list = json.load(f)
        feedback_list.append(data)
        f.seek(0)
        json.dump(feedback_list, f, indent=4)
        
def load_feedback_data(feedback_file, detailed_feedback_file):
    feedback_data = {'feedback': [], 'detailed_feedback': []}
    with open(feedback_file, 'r') as f:
        feedback_data['feedback'] = json.load(f)
    with open(detailed_feedback_file, 'r') as f:
        feedback_data['detailed_feedback'] = json.load(f)
    return feedback_data

def update_dqn_analysis():
    # Retrain your DQN agent here using feedback data
    # For example:
    # Load feedback data from JSON files
    feedback_data = load_feedback_data(FEEDBACK_FILE, DETAILED_FEEDBACK_FILE)
    # Retrain DQN agent using the feedback data
    dqnagent.retrain_with_feedback(feedback_data)
    # Update DQN analysis based on the retrained agent
    # For example, get the latest analysis from the retrained agent
    latest_analysis = dqnagent.get_latest_dqn_analysis()
    # Save the latest analysis to a file or update it in a database
    #save_latest_analysis(latest_analysis)
    


def clear_log_file(file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('')
        return True
    except Exception as e:
        print(f"Error clearing log file {file_path}: {e}")
        return False

def update_db_log(user, log_type):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            if log_type == 'keylogs':
                cursor.execute('UPDATE user_logs SET keystroke_log = "" WHERE user = ?', (user,))
            elif log_type == 'mouselogs':
                cursor.execute('UPDATE user_logs SET window_log = "" WHERE user = ?', (user,))
            elif log_type == 'camlogs':
                cursor.execute('UPDATE user_logs SET cam_log = "" WHERE user = ?', (user,))
            conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return False

@app.route('/clear/<log_type>/<user>', methods=['DELETE'])
def clear_log(log_type, user):
    log_files = {
        'keylogs': f'{os.getenv("APPDATA")}/KeystrokeLogger/{user}_keystroke_log.txt',
        'mouselogs': f'{os.getenv("APPDATA")}/KeystrokeLogger/{user}_window_log.txt',
        'camlogs': f'{os.getenv("APPDATA")}/KeystrokeLogger/{user}_webcam_log.txt'
    }
    
    if log_type not in log_files:
        return jsonify({'error': 'Invalid log type'}), 400
    
    log_file = log_files[log_type]
    
    if clear_log_file(log_file) and update_db_log(user, log_type):
        return jsonify({'success': True}), 200
    else:
        return jsonify({'error': 'Failed to clear log'}), 500

def open_browser():
    # Open the default web browser with the URL of your Flask app
    webbrowser.open_new("http://127.0.0.1:5000")

@app.route('/audio')
def get_audio():
    # Specify the path to the audio file
    audio_file_path = 'laser-cannon-science-fiction-sound-9831-[AudioTrimmer.com].mp3'
    
    # Send the audio file as a response
    return send_file(audio_file_path, mimetype='audio/mp3')

if __name__ == "__main__":
    Timer(2, open_browser).start()
    app.run(debug=True, use_reloader=False)

