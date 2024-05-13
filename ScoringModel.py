import re
from datetime import datetime
import numpy as np

def parse_log_line(line):
    try:
        if "Clipboard:" in line:
            parts = line.split(" - Clipboard: ")
            timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            action = "Clipboard"
            context = parts[1] if len(parts) > 1 else "Content unknown"
        elif "Switched to:" in line:
            parts = line.split(" - Switched to: ")
            timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            action = "Switched"
            context = parts[1] if len(parts) > 1 else "Destination unknown"
        elif ", Position:" in line:
            parts = line.split(", ")
            timestamp = datetime.strptime(parts[0].split("Timestamp: ")[1], "%Y%m%d%H%M%S")
            action = parts[1].split("Position: ")[1]
            context = parts[2].split("Proof Frame: ")[1]
        else:
            return None  # Ignore lines that do not match expected formats

        if " - Switched to: ChatGPT" in line:
            action = "Switched to ChatGPT"
        elif " - Switched to: Google" in line or " - Switched to: Chrome" in line or " - Switched to: Opera" in line:
            action = "Switched to Browser"
        elif "Looking right" in action or "Looking left" in action:
            action = "Looking"

        return {'timestamp': timestamp, 'action': action, 'context': context}
        
    except Exception as e:
        print(f"Error parsing line: {line}. Error: {e}")
        return None

def calculate_cheating_score(log_entries):
    score = 0
    for entry in log_entries:
        if entry:
            action = entry['action']  # Access dictionary by keys
            if "Alt+Tab" in action:
                score += 10
            if any(key in action for key in ["ctrl+c", "ctrl+a", "ctrl+v", "ctrl+win+right", "ctrl+win+left"]):
                score += 10
            if "Clipboard" in action:
                score += 5
            if any(alert in action for alert in ["No Face", "Using Mobile", "Looking down", "Mobile Usage", "Looking"]):
                score += 5
            if "Switched to Browser" in action:
                score += 10
            if "Switched to ChatGPT" in action:
                score += 50
    return score

def process_logs():
    try:
        # Assuming this might raise an exception
        log_files = ["./Keylogger/keylogger.txt", "./mousemovement/windowtablogger.txt", "./Eye-Tracker/webcam_log.txt"]
        all_entries = []
        for file_name in log_files:
            with open(file_name, 'r') as file:
                for line in file:
                    parsed_line = parse_log_line(line.strip())
                    if parsed_line:
                        all_entries.append(parsed_line)
        total_score = calculate_cheating_score(all_entries)
        return all_entries, total_score
    except Exception as e:
        print(f"Error processing logs: {str(e)}")
        # Return empty values or defaults if there is an error
        return [], 0


def main():
    log_files = ["./Keylogger/keylogger.txt", "./mousemovement/windowtablogger.txt", "./Eye-Tracker/webcam_log.txt"]
    all_entries = []
    
    for file_name in log_files:
        with open(file_name, 'r') as file:
            for line in file:
                parsed_line = parse_log_line(line.strip())
                if parsed_line:
                    all_entries.append(parsed_line)

    # Calculate the total cheating score
    total_score = calculate_cheating_score(all_entries)
    print(f"Total Possible Cheating Score: {total_score}")
    print("Completed processing logs without stopping on errors.")


class CheatingRiskAgent:
    def __init__(self, learning_rate=0.2, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}  # Maps scores to risk level probabilities
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_risk_level(self, score):
        if score not in self.q_table:
            self.q_table[score] = np.zeros(3)  # Assume 3 risk levels: low, medium, high
        return np.argmax(self.q_table[score])

    def update_q_table(self, score, action, reward, next_score):
        print(f"Updating Q-table for score {score}, action {action}, reward {reward}, next score {next_score}")
        next_max = np.max(self.q_table[next_score]) if next_score in self.q_table else 0
        self.q_table[score][action] += self.learning_rate * (reward + self.discount_factor * next_max - self.q_table[score][action])
        print(f"Updated Q-table: {self.q_table}")

    def feedback(self, score, action_taken, user_feedback):
        # Get the current max Q-value for future state, if available
        future_state_max_q = np.max(self.q_table[score]) if score in self.q_table else 0
        
        # Update all actions; reinforce correct action and penalize incorrect ones
        for action in range(len(self.q_table[score])):
            if action == user_feedback:
                reward = 1  # Correct action
            else:
                reward = -1  # Incorrect action
            
            # Update rule for Q-learning
            self.q_table[score][action] += self.learning_rate * (reward + self.discount_factor * future_state_max_q - self.q_table[score][action])

    def get_latest_analysis(self):
        if not self.q_table:
            return "No analysis data available."
        analysis_summary = "Risk Assessment Analysis:\n"
        for score, values in sorted(self.q_table.items()):
            risk_levels = ['Low', 'Medium', 'High']
            analysis_summary += f"Score: {score}, Risk Levels: {', '.join(f'{risk_levels[i]}: {value:.2f}' for i, value in enumerate(values))}\n"
        return analysis_summary

# Initialize agent
agent = CheatingRiskAgent()

#if __name__ == "__main__":
  #  main()
