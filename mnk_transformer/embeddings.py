import torch

def get_input_embeddings(model, sequences, device):
    print(f"Generating input embeddings for {len(sequences)} sequences")
    embeddings = []
    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print(f"On sequence {i}/{len(sequences)}")
        idx = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        tok_emb = model.transformer.wte(idx)  # (1, t, n_embd)
        pos_emb = model.transformer.wpe(torch.arange(idx.size(1), device=device))  # (t, n_embd)
        embeddings.append(tok_emb + pos_emb)  # Combined input embedding
    return embeddings  # List of tensors [(1, t, n_embd), ...]

def get_block_activations(model, sequences, device, layer=-1):
    """Extract activations from a specific transformer layer."""
    print(f"Generating activations for {len(sequences)} sequences")
    activations = []
    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print(f"On sequence {i}/{len(sequences)}")
        idx = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        with torch.no_grad():
            x = model.transformer.wte(idx) + model.transformer.wpe(torch.arange(idx.size(1), device=device))
            for i, block in enumerate(model.transformer.h):
                x = block(x)
                if i == layer:  # Capture activation at this layer
                    activations.append(x)
                    break  # Stop early if desired
    return activations  # List of tensors [(1, t, n_embd), ...]

def get_final_embeddings(model, sequences, device):
    print(f"Generating final embeddings for {len(sequences)} sequences")
    final_embeddings = []
    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print(f"On sequence {i}/{len(sequences)}")
        idx = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        with torch.no_grad():
            _, t = idx.shape
            assert t <= model.block_size, f"Seq length {t} exceeds block size {model.block_size}"
            
            pos = torch.arange(t, dtype=torch.long, device=device)  # (t,)
            x = model.transformer.wte(idx) + model.transformer.wpe(pos)
            x = model.transformer.drop(x)
            for block in model.transformer.h:
                x = block(x)
            x = model.transformer.ln_f(x)  # Final layer norm

            final_embeddings.append(x)  # (1, t, n_embd)
    return final_embeddings

def get_layerwise_embeddings(model, sequences, device):
    """Extract embeddings after each layer and return them as a list."""
    num_layers = len(model.transformer.h)  # Get the number of transformer layers
    print(f"Model has {num_layers} transformer layers, {num_layers + 2} total activations")

    layerwise_embeddings = [[] for _ in range(num_layers + 2)]  # +1 for input, +1 for final embeddings

    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print(f"On sequence {i}/{len(sequences)}")

        idx = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        with torch.no_grad():
            _, t = idx.shape
            assert t <= model.block_size, f"Seq length {t} exceeds block size {model.block_size}"

            pos = torch.arange(t, dtype=torch.long, device=device)  # (t,)
            x = model.transformer.wte(idx) + model.transformer.wpe(pos)
            layerwise_embeddings[0].append(x)  # Store input embeddings

            x = model.transformer.drop(x)
            for layer_idx, block in enumerate(model.transformer.h):
                x = block(x)
                layerwise_embeddings[layer_idx + 1].append(x)  # Store embeddings after each layer

            x = model.transformer.ln_f(x)  # Apply final layer norm
            layerwise_embeddings[num_layers + 1].append(x)  # Store final embeddings

    return layerwise_embeddings  # Now includes input, all layers, and final embeddings

def get_headwise_activations(model, sequences, device, layer=0):
    """Extract per-head attention outputs from a specific transformer layer.

    Returns:
        Tuple of lists:
            head_0_activations: list of tensors of shape (t, head_dim)
            head_1_activations: list of tensors of shape (t, head_dim)
    """
    assert model.config.n_head == 2, "Function assumes the model has exactly 2 attention heads."
    print(f"Extracting head-wise activations from layer {layer} for {len(sequences)} sequences")

    head_0_activations = []
    head_1_activations = []

    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print(f"On sequence {i}/{len(sequences)}")

        idx = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        with torch.no_grad():
            t = idx.size(1)
            assert t <= model.block_size, f"Seq length {t} exceeds block size {model.block_size}"
            x = model.transformer.wte(idx) + model.transformer.wpe(torch.arange(t, device=device))
            x = model.transformer.drop(x)

            for i, block in enumerate(model.transformer.h):
                if i == layer:
                    x_norm = block.ln_1(x)
                    per_head = block.attn(x_norm, return_head_outputs=True)  # (1, t, 2, head_dim)
                    head_0_activations.append(per_head[0, -1, 0, :].cpu())  # (head_dim,)
                    head_1_activations.append(per_head[0, -1, 1, :].cpu())  # (head_dim,)
                    break
                else:
                    x = block(x)

    return head_0_activations, head_1_activations