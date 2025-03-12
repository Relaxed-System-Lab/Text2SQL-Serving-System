import os
import json
import re

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    steps = []
    step_pattern = re.compile(r'##############################\s*(Human|AI) at step (.*?)\s*##############################')
    token_pattern = re.compile(r'The token count is:\s*(\d+)')
    
    headers = list(step_pattern.finditer(log_content))
    
    for i, header in enumerate(headers):
        role = header.group(1)
        step_name = header.group(2).strip()
        
        start_index = header.end()
        end_index = headers[i+1].start() if i + 1 < len(headers) else len(log_content)
        section_text = log_content[start_index:end_index]
        
        token_match = token_pattern.search(section_text)
        tokens = int(token_match.group(1)) if token_match else 0
        
        if role == "Human":
            steps.append({
                "step": step_name,
                "input_length": tokens,
                "output_length": 0
            })
        else:  # role == "AI"
            if steps and steps[-1]["step"] == step_name and steps[-1]["output_length"] == 0:
                steps[-1]["output_length"] = tokens
            else:
                steps.append({
                    "step": step_name,
                    "input_length": 0,
                    "output_length": tokens
                })
    
    return steps

def collect_logs(logs_directory):
    logs_data = []
    
    for log_file in os.listdir(logs_directory):
        if log_file.endswith(".log"):
            log_file_path = os.path.join(logs_directory, log_file)
            steps = parse_log_file(log_file_path)
            logs_data.extend(steps)
    
    return logs_data

def main():
    logs_directory = './results/dev/CHESS_IR_CG_UT/mini_dev/2025-03-09T12:50:35.257552/logs'
    logs_data = collect_logs(logs_directory)
    
    output_file_path = './input_file.json'
    with open(output_file_path, 'w') as output_file:
        json.dump(logs_data, output_file, indent=4)

if __name__ == "__main__":
    main()