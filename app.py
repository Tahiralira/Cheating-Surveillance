from flask import Flask, render_template
from Keylogger.keyanalyzer import analyze_keystrokes

app = Flask(__name__)

@app.route('/')
def index():
    # Run your Python script and get the output
    output = analyze_keystrokes()  # Replace with your actual function
    
    # Pass the output to an HTML template
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
