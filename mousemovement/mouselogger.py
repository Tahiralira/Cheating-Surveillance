import time
import pygetwindow as gw
import pyperclip
import threading
previous_clipboard_content = None

def check_clipboard():
    global previous_clipboard_content
    
    while True:
        clipboard_content = pyperclip.paste()
        if clipboard_content != previous_clipboard_content:
            #print("New clipboard content:", clipboard_content)
            previous_clipboard_content = clipboard_content
            # Log the clipboard content
            with open("windowtablogger.txt", "a") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Clipboard: {clipboard_content}\n")
        
        time.sleep(1)  # Adjust the polling interval as needed

def log_active_window_change():
    previous_active_window = None
    
    while True:
        active_window = gw.getActiveWindow()
        if active_window != previous_active_window:
            if active_window is not None:
                #print(f"Switched to: {active_window.title}")
                # Log the window title
                with open("windowtablogger.txt", "a") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Switched to: {active_window.title}\n")
            previous_active_window = active_window
        time.sleep(1)  # Adjust the polling interval as needed

if __name__ == "__main__":
    
    # Add space and "New Session" heading to log file
    with open("windowtablogger.txt", "a") as log_file:
        log_file.write("\n\n\n\nNew Session Log\n\n")
    
    # Start monitoring clipboard changes
    clipboard_thread = threading.Thread(target=check_clipboard)
    clipboard_thread.daemon = True
    clipboard_thread.start()
    
    # Start monitoring window changes
    log_active_window_change()
