import csv
import subprocess
import time
from datetime import datetime
import re

# Hardcoded hyperparameter combinations
combinations = [
    # (n_layer, n_head, n_embd)
    (1, 1, 4),
    (1, 1, 8),
    (1, 1, 48),
    (1, 1, 12),  # Example combination; add more as needed
    (1, 2, 12),
    (1, 3, 12),
    (1, 4, 12),
    (1, 6, 12),
    (2, 1, 8),
    (2, 2, 8),
    (2, 1, 9),
    (2, 3, 9),
]

# CSV file to store results
csv_file = 'experiment_results/dec1_arch_exp.csv'
header = ['timestamp', 'n_layer', 'n_head', 'n_embd', 'num_params', 'win_rate', 'invalid_rate', 'train_time', 'wandb_link']

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1b\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]')
    return ansi_escape.sub('', text)

def train_model(n_layer, n_head, n_embd):
    """Trains the model with the specified hyperparameters and returns training time."""
    start_time = time.time()
    result = subprocess.run(
        ['python3', 'train.py', 
         '--n_layer', str(n_layer), 
         '--n_head', str(n_head), 
         '--n_embd', str(n_embd)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    train_time = time.time() - start_time
    
    stdout_lines = result.stdout.splitlines()
    wandb_line = next((line for line in stdout_lines if "at: " in line), None)
    wandb_link = str(wandb_line.split(": ")[2]) if wandb_line else ""
    wandb_link = remove_ansi_escape_sequences(wandb_link)

    return train_time, wandb_link

def evaluate_model():
    """Evaluates the model and returns `num_params`, `win_rate`, and `invalid_rate`."""
    result = subprocess.run(
        ['python3', 'benchmark.py'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Benchmarking failed: {result.stderr}")

    # Parse the output
    output_lines = result.stdout.splitlines()
    num_params = int(output_lines[0])
    counts = eval(output_lines[1])
    win_rate = float(output_lines[2])
    invalid_rate = float(output_lines[3])
    return num_params, win_rate, invalid_rate

# Run training and evaluation, then append results to the CSV
with open(csv_file, 'a', newline='') as f:  # 'a' mode to append instead of overwrite
    writer = csv.writer(f)
    
    # Write header if file is empty
    f.seek(0, 2)
    if f.tell() == 0:
        writer.writerow(header)

    for n_layer, n_head, n_embd in combinations:
        for run in range(1):
            print(f"Running with n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
            
            # Record start timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Train and record the time taken
            train_time, wandb_link = train_model(n_layer, n_head, n_embd)
            
            # Evaluate and get metrics
            num_params, win_rate, invalid_rate = evaluate_model()
            
            # Append results to CSV
            writer.writerow([timestamp, n_layer, n_head, n_embd, num_params, win_rate, invalid_rate, train_time, wandb_link])
            print(f"Completed for n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}: "
                f"num_params={num_params}, win_rate={win_rate}, invalid_rate={invalid_rate}, "
                f"train_time={train_time:.2f} seconds, timestamp={timestamp}")