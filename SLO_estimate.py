import os
import re
import glob
from collections import defaultdict

def process_logs(log_directory):
    # Dictionaries to accumulate values per step.
    # For human, we store input token counts.
    human_tokens = defaultdict(list)
    # For AI, we store both time cost and output token counts.
    ai_time_costs = defaultdict(list)
    ai_tokens = defaultdict(list)
    
    # List all .log files in the directory.
    log_files = glob.glob(os.path.join(log_directory, "*.log"))
    
    # Process each log file.
    for filepath in log_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Use regex to locate section headers.
        # Example header: "############################## Human at step Information Retriever ##############################"
        header_pattern = r"##############################\s*(Human|AI) at step (.*?)\s*##############################"
        headers = list(re.finditer(header_pattern, content))
        
        # Iterate over each header to process its associated section.
        for i, header in enumerate(headers):
            role = header.group(1)  # "Human" or "AI"
            step = header.group(2).strip()  # the step name
            
            # Define the section content from current header to the next header (or end of file)
            start_index = header.end()
            end_index = headers[i+1].start() if i + 1 < len(headers) else len(content)
            section_text = content[start_index:end_index]
            
            if role == "Human":
                # Look for the token count line (e.g., "######The token count is: 338######")
                token_match = re.search(r"The token count is:\s*(\d+)", section_text)
                if token_match:
                    tokens = int(token_match.group(1))
                    human_tokens[step].append(tokens)
            else:  # role == "AI"
                # Extract the time cost (e.g., "######The time cost is: 8.101889848709106######")
                time_match = re.search(r"The time cost is:\s*([\d\.]+)", section_text)
                if time_match:
                    time_cost = float(time_match.group(1))
                    ai_time_costs[step].append(time_cost)
                # Extract the output token count.
                token_match = re.search(r"The token count is:\s*(\d+)", section_text)
                if token_match:
                    tokens = int(token_match.group(1))
                    ai_tokens[step].append(tokens)
    
    # Compute averages for each step.
    avg_human_tokens = {step: sum(tokens) / len(tokens) for step, tokens in human_tokens.items() if tokens}
    avg_ai_time = {step: sum(times) / len(times) for step, times in ai_time_costs.items() if times}
    avg_ai_tokens = {step: sum(tokens) / len(tokens) for step, tokens in ai_tokens.items() if tokens}
    
    return avg_human_tokens, avg_ai_time, avg_ai_tokens

def main():
    # Set the directory that contains the log files.
    log_directory = "./results/dev/CHESS_IR_CG_UT/mini_dev/2025-03-09T12:50:35.257552/logs"
    
    avg_human_tokens, avg_ai_time, avg_ai_tokens = process_logs(log_directory)
    
    # Write the averages to a file.
    output_file = "averages.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Average Human Input Tokens per Step:\n")
        for step, avg in avg_human_tokens.items():
            f.write(f"Step: {step}, Average Tokens: {avg:.2f}\n")
        
        f.write("\nAverage AI Output Tokens and Time Cost per Step:\n")
        for step in sorted(avg_ai_time.keys()):
            avg_time = avg_ai_time[step]
            avg_tokens = avg_ai_tokens.get(step, 0)
            f.write(f"Step: {step}, Average Time Cost: {avg_time:.2f}, Average Tokens: {avg_tokens:.2f}\n")
    
    print(f"Averages have been written to {output_file}")
    
if __name__ == "__main__":
    main()
