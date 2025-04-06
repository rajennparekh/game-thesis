import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from setup import load_from_checkpoint, device
from embeddings import get_layerwise_embeddings
from board_ops import check_winner

class BoardStateClassifier(nn.Module):
    def __init__(self, n_embd, board_size, task3):
        output_dim = 2 if task3 else board_size * 3
        super(BoardStateClassifier, self).__init__()
        self.fc1 = nn.Linear(n_embd, 12)
        self.fc2 = nn.Linear(12, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class LargeBoardStateClassifier(nn.Module):
    def __init__(self, n_embd, board_size, task3):
        output_dim = 2 if task3 else board_size * 3
        super(LargeBoardStateClassifier, self).__init__()
        self.fc1 = nn.Linear(n_embd, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class LinearBoardStateClassifier(nn.Module):
    def __init__(self, n_embd, board_size, task3):
        output_dim = 2 if task3 else board_size * 3
        super(LinearBoardStateClassifier, self).__init__()
        self.fc = nn.Linear(n_embd, output_dim)

    def forward(self, x):
        return self.fc(x)

def board_from_game_seq(seq, m=3, n=3):
    no_pad_seq = seq[1:]
    board = np.zeros((m, n))
    for i, s in enumerate(no_pad_seq):
        if s < m * n:
            row, col = divmod(s, n)
            board[row][col] = 1 if i % 2 == 0 else -1
    return board

def player_based_board_from_seq(seq, m=3, n=3):
    no_pad_seq = seq[1:]
    board = np.zeros((m, n))
    current_player = 1 if len(no_pad_seq) % 2 == 0 else -1
    for s in no_pad_seq:
        if s < m * n:
            row, col = divmod(s, n)
            board[row][col] = current_player
            current_player *= -1
    return board

def occupied_board_from_seq(seq, m=3, n=3):
    no_pad_seq = seq[1:]
    board = np.zeros((m, n))
    for s in no_pad_seq:
        if s < m * n:
            row, col = divmod(s, n)
            board[row][col] = 1
    return board

def is_winner_from_game_seq(seq, m=3, n=3):
    board = player_based_board_from_seq(seq, m, n)
    return [1] if check_winner(board, 3) is not None else [0]

def generate_dataset(task, df, m=3, n=3):
    dataset = []
    for game_seq in df.values.tolist():
        for i in range(1, len(game_seq) + 1):
            trimmed_seq = game_seq[:i]
            if task == 0:
                board = player_based_board_from_seq(trimmed_seq, m, n)
            elif task == 1:
                board = board_from_game_seq(trimmed_seq, m, n)
            elif task == 2:
                board = occupied_board_from_seq(trimmed_seq, m, n)
            elif task == 3:
                board = is_winner_from_game_seq(trimmed_seq, m, n)
            dataset.append((trimmed_seq, board))
    return dataset

def map_board_state(state):
    return {0: 0, 1: 1, -1: 2}.get(state, state)

def train(model, embeddings, targets, task, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(embeddings, targets), batch_size=256, shuffle=True)

    model.to(device)
    losses = []
    for _ in range(epochs):
        model.train()
        total_loss = 0
        for emb, tgt in data_loader:
            emb, tgt = emb.to(device), tgt.to(device)
            logits = model(emb)
            loss = criterion(logits, tgt) if task == 3 else criterion(logits.view(-1, 3), tgt.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(data_loader))
    return losses

def evaluate_model_accuracy(model, embeddings, targets, task, m=3, n=3):
    model.eval()
    correct_board, total_board = 0, 0
    correct_space, total_space = 0, 0
    with torch.no_grad():
        for i in range(len(embeddings)):
            x = embeddings[i].unsqueeze(0).to(device)
            logits = model(x)
            if task == 3:
                pred = torch.argmax(logits, dim=-1).item()
                true = targets[i].item()
                correct_board += int(pred == true)
                total_board += 1
            else:
                logits = logits.view(1, m * n, 3)
                pred = torch.argmax(logits, dim=-1)
                true = targets[i].to(device)
                correct_board += int(torch.equal(pred.squeeze(), true))
                correct_space += (pred.squeeze() == true).sum().item()
                total_board += 1
                total_space += true.numel()
    return correct_board / total_board, (correct_space / total_space if total_space > 0 else 0)

def run_probe_experiment(task, m, n, k, out_dir='out', name='ckpt.pt'):
    task3 = task == 3
    data = np.load(os.path.join('data', f"train_m{m}_n{n}_k{k}.npy")).astype(np.int64)
    df = pd.DataFrame(data)
    dataset = generate_dataset(task, df, m, n)
    np.random.shuffle(dataset)

    train_dataset = dataset[:100000]
    test_dataset = dataset[100000:120000]

    train_seqs = [seq for seq, _ in train_dataset]
    test_seqs = [seq for seq, _ in test_dataset]

    if task3:
        train_targets = torch.tensor([label for _, label in train_dataset], dtype=torch.long).squeeze(1).to(device)
        test_targets = torch.tensor([label for _, label in test_dataset], dtype=torch.long).squeeze(1).to(device)
    else:
        train_states = [list(map(map_board_state, board.flatten())) for _, board in train_dataset]
        test_states = [list(map(map_board_state, board.flatten())) for _, board in test_dataset]
        train_targets = torch.tensor(train_states, dtype=torch.long).to(device)
        test_targets = torch.tensor(test_states, dtype=torch.long).to(device)

    model = load_from_checkpoint(out_dir, name).to(device)
    model.eval()

    train_embs = get_layerwise_embeddings(model, train_seqs, device)
    test_embs = get_layerwise_embeddings(model, test_seqs, device)

    layer_num_to_results = {}
    for i, (train_layer, test_layer) in enumerate(zip(train_embs, test_embs)):
        print(f"On layer {i + 1}")
        train_x = torch.stack([act[:, -1, :] for act in train_layer]).squeeze(1)
        test_x = torch.stack([act[:, -1, :] for act in test_layer]).squeeze(1)

        model_mlp = BoardStateClassifier(train_x.size(1), m * n, task3)
        model_linear = LinearBoardStateClassifier(train_x.size(1), m * n, task3)
        model_mlp_large = LargeBoardStateClassifier(train_x.size(1), m * n, task3)
        models = [(model_mlp, "small mlp"), (model_linear, "linear"), (model_mlp_large, "large mlp")]

        loss_dict, board_acc_dict, space_acc_dict = {}, {}, {}
        for model, name in models:
            print(f"Training {name}")
            losses = train(model, train_x, train_targets, task, epochs=40)
            board_acc, space_acc = evaluate_model_accuracy(model, test_x, test_targets, task, m, n)
            loss_dict[name] = losses
            board_acc_dict[name] = board_acc
            space_acc_dict[name] = space_acc

        layer_num_to_results[i] = (
            loss_dict,
            board_acc_dict,
            space_acc_dict,
            model_mlp,
            model_mlp_large,
            model_linear,
            train_x,
            test_x
        )

    return layer_num_to_results, train_targets, test_targets