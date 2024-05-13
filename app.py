# @app.route('/images/<path:filename>')
# def send_image(filename):
#     return send_from_directory('Eye-Tracker', filename)


from flask import Flask, jsonify, send_from_directory, render_template, request, session, url_for, redirect
import os
from flask_cors import CORS
import ScoringModel
import numpy as np
from ScoringModel import CheatingRiskAgent
import sqlite3
from flask import g

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

@app.route('/keylogs/<username>')
def get_key_logs(username):
    try:
        # Fetch keystroke logs for the selected user from the database
        cursor = get_db().cursor()  # Define the cursor variable
        cursor.execute("SELECT keystroke_log FROM user_logs WHERE user=?", (username,))
        logs = cursor.fetchone()[0]
        return jsonify({'data': logs})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/mouselogs/<username>')
def get_mouse_logs(username):
    try:
        # Fetch keystroke logs for the selected user from the database
        cursor = get_db().cursor()  # Define the cursor variable
        cursor.execute("SELECT window_tab_log FROM user_logs WHERE user=?", (username,))
        logs = cursor.fetchone()[0]
        return jsonify({'data': logs})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/camlogs/<username>')
def get_cam_logs(username):
    try:
        # Fetch keystroke logs for the selected user from the database
        cursor = get_db().cursor()  # Define the cursor variable
        cursor.execute("SELECT cam_log FROM user_logs WHERE user=?", (username,))
        logs = cursor.fetchone()[0]
        return jsonify({'data': logs})
    except Exception as e:
        return jsonify({'error': str(e)})

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

if __name__ == "__main__":
    app.run(debug=True)


