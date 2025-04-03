import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from setup import load_from_checkpoint, device
from embeddings import get_layerwise_embeddings

#TODO: Can we do further analysis? Confusion matrix results?

class BoardStateClassifier(nn.Module):
    def __init__(self, n_embd, board_size):
        super(BoardStateClassifier, self).__init__()
        self.fc1 = nn.Linear(n_embd, 12)  # First hidden layer
        # self.fc2 = nn.Linear(12, 12)      # Second hidden layer
        self.fc2 = nn.Linear(12, board_size * 3)  # Output layer: 3 classes per board space
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        # x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc2(x)  # No activation here, we'll apply softmax later
        return x  # logits
    
class LargeBoardStateClassifier(nn.Module):
    def __init__(self, n_embd, board_size):
        super(LargeBoardStateClassifier, self).__init__()
        self.fc1 = nn.Linear(n_embd, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 16)      # Second hidden layer
        self.fc3 = nn.Linear(16, board_size * 3)  # Output layer: 3 classes per board space
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # No activation here, we'll apply softmax later
        return x  # logits
    
class LinearBoardStateClassifier(nn.Module):
    def __init__(self, n_embd, board_size):
        super(LinearBoardStateClassifier, self).__init__()
        self.fc = nn.Linear(n_embd, board_size * 3)  # Direct mapping from input to output

    def forward(self, x):
        return self.fc(x)
    
def board_from_game_seq(seq, m=3, n=3):
    no_pad_seq = seq[1:]
    board = np.zeros((m, n))
    for i in range(len(no_pad_seq)):
        s = no_pad_seq[i]
        if s < m * n:
            row, col = s // n, s % n
            if i % 2 == 0:
                board[row][col] = 1
            else:
                board[row][col] = -1
    return board

def player_based_board_from_seq(seq, m=3, n=3):
    no_pad_seq = seq[1:]
    board = np.zeros((m, n))
    current_player = 1 if len(no_pad_seq) % 2 == 0 else -1
    
    for i, s in enumerate(no_pad_seq):
        if s < m * n:
            row, col = s // n, s % n
            board[row][col] = current_player
            current_player *= -1  # Switch perspective after each move
    return board

def occupied_board_from_seq(seq, m=3, n=3):
    no_pad_seq = seq[1:]
    board = np.zeros((m, n))
    for s in no_pad_seq:
        if s < m * n:
            row, col = s // n, s % n
            board[row][col] = 1
    return board

def generate_dataset(task, df, m=3, n=3):
    dataset = []
    game_seqs = df.values.tolist()
    for game_seq in game_seqs:
        for i in range(1, len(game_seq) + 1):
            current_board = []
            trimmed_seq = game_seq[:i]
            if task == 0:
                current_board = player_based_board_from_seq(trimmed_seq, m, n)
            elif task == 1:
                current_board = board_from_game_seq(trimmed_seq, m, n)
            elif task == 2:
                current_board = occupied_board_from_seq(trimmed_seq, m, n)
            dataset.append((trimmed_seq, current_board))
    return dataset

def map_board_state(state):
    return {0: 0, 1: 1, -1: 2}.get(state, state)  # default to state if not in dict

def train(model, embeddings_tensor_sq, board_states_tensor, epochs=20):
    losses = []
    model.to(device)  # Move the model to the appropriate device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 256
    dataset_tensor = torch.utils.data.TensorDataset(embeddings_tensor_sq, board_states_tensor)
    data_loader = torch.utils.data.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode at start of each epoch
        total_loss = 0
        for embeddings, targets in data_loader:
            embeddings, targets = embeddings.to(device), targets.to(device)

            logits = model(embeddings)
            loss = criterion(logits.view(-1, 3), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss / len(data_loader))

    return losses


def evaluate_model_accuracy(model, final_embeddings_tensor_sq, board_states_tensor, device, m=3, n=3):
    model.eval()  # Set the model to evaluation mode

    correct_board_predictions = 0
    total_board_predictions = 0
    correct_space_predictions = 0
    total_space_predictions = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for index in range(len(final_embeddings_tensor_sq)):
            # Select one sample from final_embeddings_tensor_sq
            sample_embedding = final_embeddings_tensor_sq[index].unsqueeze(0).to(device)  # Add batch dimension

            # Pass the sample through the model
            logits = model(sample_embedding)  # Shape: [1, m * n * 3]

            # Reshape the logits to [1, m * n, 3]
            logits = logits.view(1, m * n, 3)

            # Get the predicted classes by applying argmax on the last dimension (dim=-1)
            predicted_classes = torch.argmax(logits, dim=-1)  # Shape: [1, m * n]

            # Get the ground truth board state for the current index and move it to the same device
            true_board_state = board_states_tensor[index].to(device)

            # Compare the predicted classes with the ground truth board state
            if torch.equal(predicted_classes.squeeze(), true_board_state):
                correct_board_predictions += 1

            total_board_predictions += 1

            correct_space_predictions += (predicted_classes.squeeze() == true_board_state).sum().item()
            total_space_predictions += true_board_state.numel()

    # Calculate accuracy: Number of correct predictions / Total predictions
    board_accuracy = correct_board_predictions / total_board_predictions
    space_accuracy = correct_space_predictions / total_space_predictions

    return board_accuracy, space_accuracy

def run_probe_experiment(task, m, n, k, out_dir='out', name='ckpt.pt'):
    data_dir = "data"
    train_data = np.load(os.path.join(data_dir, f"train_m{m}_n{n}_k{k}.npy")).astype(dtype=np.int64)
    df = pd.DataFrame(train_data)
    
    dataset = generate_dataset(task, df)
    np.random.shuffle(dataset)  # Shuffle dataset before selecting samples

    # Select fixed-size train and test datasets
    train_dataset = dataset[:100000]
    test_dataset = dataset[100000:120000]

    # Extract sequences and board states for train and test sets
    train_sequences = [seq for seq, _ in train_dataset]
    train_board_states = np.array([board.flatten() for _, board in train_dataset])

    test_sequences = [seq for seq, _ in test_dataset]
    test_board_states = np.array([board.flatten() for _, board in test_dataset])

    # Convert board states to tensor format (train & test)
    train_mapped_board_states = [list(map(map_board_state, board)) for board in train_board_states]
    train_board_states_tensor = torch.tensor(train_mapped_board_states, dtype=torch.long).to(device)

    test_mapped_board_states = [list(map(map_board_state, board)) for board in test_board_states]
    test_board_states_tensor = torch.tensor(test_mapped_board_states, dtype=torch.long).to(device)

    # Get activations from every layer of the model
    model = load_from_checkpoint(out_dir, name)
    model.eval()
    model.to(device)

    train_activations_list = get_layerwise_embeddings(model, train_sequences, device)
    test_activations_list = get_layerwise_embeddings(model, test_sequences, device)

    layer_num_to_results = {}

    for i, (train_activations, test_activations) in enumerate(zip(train_activations_list, test_activations_list)):
        print(f"On layer number {i + 1}")
        # Extract last-token activations
        train_activations_last_token = [activation[:, -1, :] for activation in train_activations]
        train_activations_tensor = torch.stack(train_activations_last_token).squeeze(1)

        test_activations_last_token = [activation[:, -1, :] for activation in test_activations]
        test_activations_tensor = torch.stack(test_activations_last_token).squeeze(1)

        # Initialize models
        model_mlp = BoardStateClassifier(n_embd=train_activations_tensor.size(1), board_size=m * n)
        model_linear = LinearBoardStateClassifier(n_embd=train_activations_tensor.size(1), board_size=m * n)
        model_mlp_large = LargeBoardStateClassifier(n_embd=train_activations_tensor.size(1), board_size=m * n)
        models = [(model_mlp, "small mlp"), (model_linear, "linear"), (model_mlp_large, "large mlp")]

        loss_by_model, board_accuracy_by_model, space_accuracy_by_model = {}, {}, {}

        for model, model_name in models:
            # Train on the training set
            print(f"Training {model_name}")
            losses = train(model, train_activations_tensor, train_board_states_tensor, epochs=40)
            loss_by_model[model_name] = losses

            # Evaluate on the test set
            board_accuracy, space_accuracy = evaluate_model_accuracy(model, test_activations_tensor, test_board_states_tensor, device, m, n)
            board_accuracy_by_model[model_name] = board_accuracy
            space_accuracy_by_model[model_name] = space_accuracy
        
        layer_num_to_results[i] = (loss_by_model, 
                                   board_accuracy_by_model, 
                                   space_accuracy_by_model, 
                                   model_mlp,
                                   model_mlp_large, 
                                   model_linear, 
                                   train_activations_tensor,
                                   test_activations_tensor)

    return (layer_num_to_results, train_board_states_tensor, test_board_states_tensor)