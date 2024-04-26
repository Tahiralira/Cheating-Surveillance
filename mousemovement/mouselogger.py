import time
import pygetwindow as gw
from pynput.mouse import Listener, Button

# Global variables to track mouse actions
is_selecting = False
is_copying = False

def on_click(x, y, button, pressed):
    global is_selecting, is_copying
    
    if button == Button.right and not pressed:
        if is_selecting:
            is_selecting = False
            print("Right-clicked to copy")
            # Log the copy action
            log_action("Copy")
    elif button == Button.left:
        if pressed:
            is_selecting = True
            print("Started selecting")
        else:
            if is_selecting:
                is_selecting = False
                print("Finished selecting")

def log_action(action):
    active_window = gw.getActiveWindow()
    if active_window is not None:
        # Log the window title and the action
        with open("action_log.txt", "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {active_window.title}: {action}\n")

def log_active_window_change():
    previous_active_window = None
    
    while True:
        active_window = gw.getActiveWindow()
        if active_window != previous_active_window:
            if active_window is not None:
                print(f"Switched to: {active_window.title}")
                # Log the window title
                with open("windowtablogger.txt", "a") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Switched to: {active_window.title}\n")
            previous_active_window = active_window
        time.sleep(1)  # Adjust the polling interval as needed

if __name__ == "__main__":
    # Start the mouse listener
    with Listener(on_click=on_click) as listener:
        # Start monitoring window changes in a separate thread
        log_active_window_change()
        listener.join()
