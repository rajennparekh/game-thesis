import os
import time

import numpy as np
import torch

from tokens import PAD
from setup import init_model, save_checkpoint

import argparse

# Set up argument parser with the arguments we can pass
parser = argparse.ArgumentParser(description='Training file')
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--n_embd', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--bias', type=bool, default=False)
parser.add_argument('--mlp_layer_mult', type=int, default=4)
parser.add_argument('--model_version', type=str, default='gpt')
parser.add_argument('--train_time', type=int, default=None, 
                    help="Training duration in seconds. Overrides max_iters if set.")

# Parse the arguments
args = parser.parse_args()

# Access the arguments to pass into model setup
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
mlp_layer_mult = args.mlp_layer_mult
model_version = args.model_version
train_time = args.train_time

end_time = None
if train_time is not None:
    end_time = time.time() + train_time

save_interval = 1000

wandb_log = True
wandb_project = "ttt"

batch_size = 2048

learning_rate = 6e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Set device to MPS if available, otherwise fall back to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

data_dir = "data"
train_data = np.load(os.path.join(data_dir, "train.npy")).astype(dtype=np.int64)

def get_batch():
    data = train_data
    ix = torch.randint(data.shape[0], (batch_size,))
    x = torch.from_numpy(data[ix, :])
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = PAD
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # Removed pin_memory
    return x, y

iter_num = 0

# initializes the model with our specified parameters/architecture
model = init_model(n_layer, n_head, n_embd, dropout, bias, 
                   mlp_layer_mult, model_version=model_version)

model.to(device)
model.train()

optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device
)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project)

# Get the first batch of data
loss_by_iter = []

X, Y = get_batch()
t0 = time.time()
while (iter_num < max_iters if train_time is None else time.time() < end_time):
    if iter_num > 0 and iter_num % save_interval == 0:
        save_checkpoint(model)
        print(iter_num)


    logits, loss = model(X, Y)

    if iter_num + 1 % 100 == 0:
        loss_by_iter.append(loss)
    
    loss.backward()

    # Get a new batch after calculating gradients
    X, Y = get_batch()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    lossf = loss.item()
    # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    if wandb_log:
        wandb.log(
            {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": learning_rate,
            }
        )

    iter_num += 1
print(f"Iters trained: {iter_num}")
save_checkpoint(model)
loss_by_iter