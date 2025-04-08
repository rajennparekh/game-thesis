import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random 

def display_random_training_run():
    output_dir = Path("final_experiments")
    files = list(output_dir.glob("param_exp_333_trial*.pkl"))

    if not files:
        print("No saved experiment files found.")
        df = pd.DataFrame()  # Initialize an empty DataFrame
    else:
        dfs = []  # List to store DataFrames
        for i, file in enumerate(files):
            dfs.append(pd.read_pickle(file))
    df = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames
    converged_df = df[df["train_loss_by_iter"].apply(lambda x: x[-1] < 1.235)]
    stalling_df = converged_df[converged_df["train_loss_by_iter"].apply(lambda x: sum(1.36 <= val <= 1.37 for val in x) >= 2)]


    sample_idx = random.choice(stalling_df.index)
    row = stalling_df.loc[sample_idx]

    # Scale val_loss_by_iter to [0, 1]
    val_loss_scaled = (np.array(row['val_loss_by_iter']) - 1.2) / (1.7 - 1.2)

    plt.figure(figsize=(10, 6))
    plt.plot(val_loss_scaled, label='Rescaled Validation Loss', linestyle='-')
    plt.plot(row['non_pad_invalid_rate'], label='Invalid Move Rate', linestyle='-')
    plt.plot(row['correct_ending_rate'], label='Correct Ending Rate', linestyle='-')

    plt.xlabel('Iteration / 5k')
    plt.ylabel('Normalized Metric Value')
    plt.title(f'Metrics Over Iterations for Sampled Run {sample_idx}')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_converging_runs():
    plotting = 'val_loss_by_iter'
    output_dir = Path("final_experiments")
    files = list(output_dir.glob("param_exp_333_trial*.pkl"))
    plt.figure(figsize=(10, 6))

    if not files:
        print("No saved experiment files found.")
        df = pd.DataFrame()  # Initialize an empty DataFrame
    else:
        dfs = []  # List to store DataFrames
        for i, file in enumerate(files):
            dfs.append(pd.read_pickle(file))
    df = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames
    converged_df = df[df["train_loss_by_iter"].apply(lambda x: x[-1] < 1.235)]
    for idx, row in converged_df.iterrows():
        plt.plot(row[plotting], label=f'Run {idx}')

    plt.xlabel('Iteration / 5k')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss by Iteration for Different Runs')
    plt.show()