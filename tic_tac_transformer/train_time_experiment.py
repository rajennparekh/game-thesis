import csv
import subprocess
import time
from datetime import datetime
import re
# Training durations to test (in seconds)
train_times = [60, 300, 1200, 3600]  # Example: 1 min, 2 min, 5 min, 10 min, 20 min

# CSV file to store results
csv_file = 'experiment_results/dec1_train_time_experiment.csv'
header = ['timestamp', 'train_time', 'num_params', 'win_rate', 'invalid_rate', 'iters', 'train_link']

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1b\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]')
    return ansi_escape.sub('', text)

def train_model(train_time):
    """
    Trains the model for a specific amount of time and returns the elapsed time and iter count.
    """
    start_time = time.time()
    result = subprocess.run(
        [
            'python3', 'train.py', 
            '--n_layer', '1', 
            '--n_head', '1', 
            '--n_embd', '12',
            '--train_time', str(train_time)
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    elapsed_time = time.time() - start_time
    
    # Parse stdout for iter count
    stdout_lines = result.stdout.splitlines()
    iter_line = next((line for line in stdout_lines if line.startswith("Iters trained:")), None)
    iters = int(iter_line.split(":")[1].strip()) if iter_line else -1  # Fallback to -1 if not found
    
    wandb_line = next((line for line in stdout_lines if "at: " in line), None)
    wandb_link = str(wandb_line.split(": ")[2]) if wandb_line else ""
    wandb_link = remove_ansi_escape_sequences(wandb_link)

    return elapsed_time, iters, wandb_link

def evaluate_model():
    """
    Evaluates the model and returns `num_params`, `win_rate`, and `invalid_rate`.
    """
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

# Run training and evaluation, then log results to a CSV
with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    
    # Write header if the file is empty
    f.seek(0, 2)  # Move to end of file
    if f.tell() == 0:
        writer.writerow(header)

    for train_time in train_times:
        for run in range(5):  # Repeat each duration multiple times
            print(f"Running training for {train_time} seconds.")
            
            # Record start timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Train the model
            elapsed_time, iters, wandb_link = train_model(train_time)
            print("Training completed")
            # Evaluate and get metrics
            num_params, win_rate, invalid_rate = evaluate_model()
            
            # Append results to CSV
            writer.writerow([timestamp, elapsed_time, num_params, win_rate, invalid_rate, iters, wandb_link])
            print(f"Completed training for {train_time} seconds: "
                  f"num_params={num_params}, win_rate={win_rate}, invalid_rate={invalid_rate}, "
                  f"elapsed_time={elapsed_time:.2f} seconds, iters={iters}, timestamp={timestamp}")