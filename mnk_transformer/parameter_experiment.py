import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from train_function import train
from generate_data import run_generation

data_name = "333_two_head" #change this

# pseudocode... for different mnk...
# 1. Check if data is generated and if not, generate it
# 2. For each combo of (n_layer, n_head, n_embd) we want to...
# 2a) Train and record 
mnk_options = [(3, 3, 3)]
layer_head_embd_options = [(1, 2, 12)]
train_trials = 20

results = []

for mnk_option in mnk_options:
    m, n, k, = mnk_option
    filepath = Path(f"data/train_m{m}_n{n}_k{k}.npy")
    if not filepath.exists():
        print("Generating training data...")
        run_generation(m, n, k)

    for lhe_option in layer_head_embd_options:
        n_layer, n_head, n_embd, = lhe_option

        for _ in range(train_trials):
            (n_params, n_iter, epml_by_iter, invalid_rate_by_iter, non_pad_invalid_by_iter,
            correct_ending_rate_by_iter, end_token_possible_by_iter, 
            creativity_rate_by_iter, original_games_possible_by_iter, 
            train_loss_by_iter, val_loss_by_iter, pre_train_probe, post_train_probe) = train(m, n, k, n_layer, n_head, n_embd, n_iter=100000, verbose=True, probe=False)

            results.append({
                "m": m, "n": n, "k": k,
                "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
                "n_params": n_params, "n_iter": n_iter,
                "train_loss_by_iter": train_loss_by_iter,
                "val_loss_by_iter": val_loss_by_iter,
                "expected_perfect_model_loss": epml_by_iter,
                "invalid_rate": invalid_rate_by_iter,
                "non_pad_invalid_rate": non_pad_invalid_by_iter,
                "correct_ending_rate": correct_ending_rate_by_iter,
                "correct_ending_opportunities": end_token_possible_by_iter,
                "creativity_rate": creativity_rate_by_iter,
                "creativity_opportunities": original_games_possible_by_iter,
                "pre_train_probe": pre_train_probe,
                "post_train_probe": post_train_probe
            })
df = pd.DataFrame(results)

output_dir = Path("final_experiments")
output_dir.mkdir(parents=True, exist_ok=True)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if data_name is None:
    output_path = output_dir / f"param_exp_{timestamp}.pkl"
    df.to_pickle(output_path)
else:
    output_path = output_dir / f"param_exp_{data_name}.pkl"
    if Path(output_path).exists():
        output_path = output_dir / f"param_exp_{data_name}_{timestamp}.pkl"
    df.to_pickle(output_path)
print(f"Saved results to {output_path}")