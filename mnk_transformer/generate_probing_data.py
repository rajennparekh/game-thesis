import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from train_function import train

out_dir = 'probing_checkpoints'

(num_params, iter_num, epml_by_iter, invalid_rate_by_iter, non_pad_invalid_by_iter,
 correct_ending_rate_by_iter, end_token_possible_by_iter,
 creativity_rate_by_iter, original_games_possible_by_iter, 
 loss_by_iter, val_loss_by_iter, pre_train_probe, post_train_probe) = train(3, 3, 3, generate_probing_data=True, output_dir=out_dir)

plt.figure(figsize=(8, 5))
plt.plot(val_loss_by_iter, label=f'Val loss')
plt.xlabel('Iteration / 5k')
plt.ylabel('Val loss')
plt.show()