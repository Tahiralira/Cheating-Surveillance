import pyautogui
import time

def log_mouse_movement(interval=1):
    while True:
        x, y = pyautogui.position()
        # Write mouse position to log file
        with open('mouse_movement_log.txt', 'a') as f:
            f.write(f"({x}, {y})\n")
        time.sleep(interval)

if __name__ == "__main__":
    log_mouse_movement()
