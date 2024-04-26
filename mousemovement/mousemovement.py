import time
import pygetwindow as gw

def log_active_window_change():
    previous_active_window = None
    
    while True:
        active_window = gw.getActiveWindow()
        if active_window != previous_active_window:
            if active_window is not None:
                print(f"Switched to: {active_window.title}")
                # Log the window title or any other relevant information
                with open("window_log.txt", "a") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Switched to: {active_window.title}\n")
            previous_active_window = active_window
        time.sleep(1)  # Adjust the polling interval as needed

if __name__ == "__main__":
    log_active_window_change()
