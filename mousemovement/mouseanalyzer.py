import re
from collections import namedtuple, defaultdict

Event = namedtuple("Event", ["tab_change_before", "clipboard_data", "tab_change_after"])

def read_log_file(file_path):
    sessions = defaultdict(list)
    current_session = None
    current_event = None

    with open(file_path, "r") as log_file:
        for line in log_file:
            line = line.strip()
            if line.startswith("New Session"):
                if current_session:  # Append the previous session if it's not empty
                    sessions[len(sessions) + 1] = current_session
                current_session = []
                current_event = None
            elif line.startswith("Switched to:"):
                window_title = re.match(r"(.+?) - Switched to: (.+)", line)
                if window_title:
                    window_title = window_title.group(2)
                    if current_event is not None and current_event.clipboard_data:
                        current_event = current_event._replace(tab_change_after=window_title)
                        current_session.append(current_event)
                        current_event = None
                    else:
                        current_event = Event(window_title, None, None)
            elif line.startswith("Clipboard:"):
                clipboard_data = re.match(r"(.+?) - Clipboard: (.+)", line)
                if clipboard_data and current_event:
                    clipboard_data = clipboard_data.group(2)
                    current_event = current_event._replace(clipboard_data=clipboard_data)
        if current_session:  # Append the last session if it's not empty
            sessions[len(sessions) + 1] = current_session

    return sessions

if __name__ == "__main__":
    log_file_path = "windowtablogger.txt"
    sessions = read_log_file(log_file_path)
    print("!")
    # Print sessions and events within each session
    for session_num, events in sessions.items():
        print(f"Session {session_num}:")
        print("!")
        for event_num, event in enumerate(events, start=1):
            print(f"  Event {event_num}:")
            print(f"    Tab Change Before: {event.tab_change_before}")
            if event.clipboard_data:
                print(f"    Clipboard Data: {event.clipboard_data}")
            else:
                print("    Clipboard Data: N/A")
            print(f"    Tab Change After: {event.tab_change_after}")
