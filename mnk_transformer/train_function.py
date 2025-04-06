import numpy as np
import torch
import os
# from tokens import PAD
from setup import init_model, save_checkpoint
from benchmark_function import benchmark
from probe_functions import run_probe_experiment

def get_batch(train_data, batch_size, m, n, device):
    pad_token = m * n + 1
    data = train_data
    ix = torch.randint(data.shape[0], (batch_size,))
    x = torch.from_numpy(data[ix, :])
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = pad_token
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # Removed pin_memory
    return x, y

def get_num_params_old(m, n, num_layer, num_embed, mlp_layer_mult, exclude=True):
    block_size = m * n + 1
    vocab_size = m * n + 2
    sum = 0
    if not exclude: # These will always be excluded
        sum = sum + vocab_size * num_embed # Word token embedding
        sum = sum + block_size * num_embed # Word position embedding
    for _ in range(num_layer): # For each layer...
        sum = sum + num_embed # LayerNorm
        sum = sum + 3 * num_embed * num_embed # SelfAttention weight matrix
        sum = sum + num_embed * num_embed # SelfAttention c_proj matrix
        sum = sum + num_embed # LayerNorm
        sum = sum + num_embed * mlp_layer_mult * num_embed # MLP weights
        sum = sum + mlp_layer_mult * num_embed * num_embed # MLP weights
    sum = sum + num_embed # LayerNorm
    sum = sum + num_embed * vocab_size # Linear layer
    return sum

def get_num_params(model_version, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in model_version.parameters())
        if non_embedding:
            n_params -= model_version.transformer.wpe.weight.numel()
        return n_params

def train(m, n, k, n_layer=1, n_head=1, n_embd=12, dropout=0.0, n_iter=100000,
          bias=False, mlp_layer_mult=4, verbose=True, val_split=0.2, probe=False,
          generate_probing_data=False, output_dir='ckpt.pt'):
    # torch.mps.empty_cache()
    # num_params = get_num_params(m, n, n_layer, n_embd, mlp_layer_mult)

    batch_size = 2048
    learning_rate = 6e-4
    max_iters = n_iter
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_dir = "data"
    data = np.load(os.path.join(data_dir,f"train_m{m}_n{n}_k{k}.npy")).astype(dtype=np.int64)

    split_idx = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split_idx], data[split_idx:]

    iter_num = 0
    model = init_model(m, n, k, n_layer, n_head, n_embd, dropout, bias, 
                   mlp_layer_mult)
    num_params = get_num_params(model)
    model.to(device)
    if generate_probing_data:
        save_checkpoint(model, out_dir=output_dir, name='ckpt1')
    model.train()
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device
    )
    save_checkpoint(model)
    if probe:
        pre_train_probe = run_probe_experiment(m, n, k)
    else:
        pre_train_probe = 0

    epml_by_iter = []
    invalid_rate_by_iter = []
    non_pad_invalid_by_iter = []
    correct_ending_rate_by_iter = []
    end_token_possible_by_iter = []
    creativity_rate_by_iter = []
    original_games_possible_by_iter = []
    loss_by_iter = []
    val_loss_by_iter = []

    reached_137 = False
    reached_134 = False

    X, Y = get_batch(train_data, batch_size, m, n, device)
    while (iter_num < max_iters):
        _, loss = model(X, Y)
        if (iter_num + 1) % 5000 == 0:
            save_checkpoint(model)
            if verbose: 
                print(iter_num)
            loss_by_iter.append(loss.item())
            (expected_perfect_loss, invalid_rate, non_pad_invalid_rate, correct_ending_rate, 
            end_token_possible, creativity_rate, original_games_possible) = benchmark(m, n, k)
            epml_by_iter.append(expected_perfect_loss)
            invalid_rate_by_iter.append(invalid_rate)
            non_pad_invalid_by_iter.append(non_pad_invalid_rate)
            correct_ending_rate_by_iter.append(correct_ending_rate)
            end_token_possible_by_iter.append(end_token_possible)
            creativity_rate_by_iter.append(creativity_rate)
            original_games_possible_by_iter.append(original_games_possible)
            model.eval()
            with torch.no_grad():
                X_val, Y_val = get_batch(val_data, batch_size, m, n, device)
                _, val_loss = model(X_val, Y_val)
                val_loss_by_iter.append(val_loss.item())
                if val_loss < 1.37 and not reached_137 and generate_probing_data:
                    save_checkpoint(model, out_dir=output_dir, name='ckpt2')
                    reached_137 = True
                if val_loss < 1.34 and not reached_134 and generate_probing_data:
                    save_checkpoint(model, out_dir=output_dir, name='ckpt3')
                    reached_134 = True
            model.train()
        # probabilities = torch.softmax(logits, dim=0)
            
        loss.backward()

        # Get a new batch after calculating gradients
        X, Y = get_batch(train_data, batch_size, m, n, device)

        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        iter_num += 1
    save_checkpoint(model)
    
    if generate_probing_data:
        save_checkpoint(model, out_dir=output_dir, name='ckpt4')
    model.train()

    if probe:
        post_train_probe = run_probe_experiment(m, n, k)
    else:
        post_train_probe = 0
    
    return (num_params, iter_num, epml_by_iter, invalid_rate_by_iter, non_pad_invalid_by_iter,
            correct_ending_rate_by_iter, end_token_possible_by_iter, 
            creativity_rate_by_iter, original_games_possible_by_iter, 
            loss_by_iter, val_loss_by_iter, pre_train_probe, post_train_probe)