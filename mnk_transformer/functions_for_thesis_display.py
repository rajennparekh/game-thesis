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
    converged_df = df[df["train_loss_by_iter"].apply(lambda x: x[-1] < 1.25)]
    stalling_df = converged_df[converged_df["train_loss_by_iter"].apply(lambda x: sum(1.35 <= val <= 1.37 for val in x) >= 6)]


    sample_idx = random.choice(stalling_df.index)
    row = stalling_df.loc[sample_idx]

    # Scale val_loss_by_iter to [0, 1]
    val_loss_scaled = (np.array(row['val_loss_by_iter']) - 1.2) / (1.7 - 1.2)

    plt.figure(figsize=(10, 6))
    plt.plot(val_loss_scaled, label='val_loss (scaled)', linestyle='-')
    plt.plot(row['non_pad_invalid_rate'], label='non_pad_invalid_rate', linestyle='-')
    plt.plot(row['correct_ending_rate'], label='correct_ending_rate', linestyle='-')

    plt.xlabel('Iteration / 5k')
    plt.ylabel('Normalized Metric Value')
    plt.title(f'Metrics Over Iterations for Sampled Run {sample_idx}')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()