import torch
# from tokens import START
from board_ops import check_winner, board_full, get_valid_moves
import numpy as np
from setup import load_from_checkpoint, device
import random
import pandas as pd
import os

def expected_loss_perfect_model(df):
    """
    Compute the expected loss of a perfect model given training data.

    Args:
        df (pd.DataFrame): DataFrame where each row is a game, each column is a move,
                           and the first column is a start token.

    Returns:
        float: The expected cross-entropy loss.
    """
    num_columns = df.shape[1]
    board_size = num_columns - 1  # Since first column is a start token

    losses = []
    for col in range(1, num_columns):  # Ignore the first column
        k = board_size - (col - 1)  # Remaining valid moves

        # Default uniform probability for non-terminal states
        # print(k)
        probabilities = np.ones(len(df)) / k  
        # Correctly guess padding moves (when df.iloc[:, col] == board_size + 1)
        mask = df.iloc[:, col] == board_size + 1
        probabilities[mask] = 1.0  # Assign probability of 1 for padding moves
        # print(probabilities)
        # print(np.mean(probabilities))
        # print(len(probabilities))
        losses.append(-np.log(probabilities))

    return np.mean(np.concatenate(losses))

def benchmark(m, n, k, n_trials=1000, silent=True):
    data_dir = "data"
    train_data = pd.DataFrame(np.load(os.path.join(data_dir,f"train_m{m}_n{n}_k{k}.npy")).astype(dtype=np.int64))
    model = load_from_checkpoint()
    model.eval()
    model.to(device)

    expected_perfect_loss = expected_loss_perfect_model(train_data)

    start_token = m * n

    with torch.no_grad():
        invalid = 0
        non_pad_invalid = 0
        non_pad_invalid_possible = n_trials
        end_token_possible = 0
        end_token_generated = 0
        original_games_possible = 0
        original_games_genrated = 0

        # Generate n games
        for _ in range(n_trials):
            board = np.zeros((m, n), dtype=int)
            player = 1
            winner = None
            moves = [start_token]
            while winner is None and not board_full(board):
                if player == 1 or player == -1:
                    x = torch.tensor(moves, dtype=torch.long, device=device)[None, ...]
                    y = model.generate(x, max_new_tokens=1, temperature=1.0, top_k=3)
                    y = y[0][-1].item()

                    if y not in set(range(m * n)) or y in moves:
                        if not silent:
                            print(f"invalid move: {y} moves: {moves}")
                        winner = None
                        invalid += 1
                        ## doing this totally wrong!
                        if y in set(range(m * n)):
                            non_pad_invalid += 1
                        else:
                            non_pad_invalid_possible -= 1
                        break

                    i, j = divmod(y, n)
                else:
                    i, j = random.choice(get_valid_moves(board))

                moves.append(i * n + j)
                board[i][j] = player
                player *= -1
                winner = check_winner(board, k)
            
            # Checking creativity
            original_games_possible += 1
            full_length_game = moves.copy()
            target_len = m * n + 1
            while len(full_length_game) < target_len:
                full_length_game.append(target_len)
            moves_series = pd.Series(full_length_game)
            if not (train_data.iloc[:, :target_len] == moves_series).all(axis=1).any():
                original_games_genrated += 1

            # Checking identification of game end 
            if winner is not None and not board_full(board):
                end_token_possible += 1
                x = torch.tensor(moves, dtype=torch.long, device=device)[None, ...]
                y = model.generate(x, max_new_tokens=1, temperature=1.0, top_k=3)
                y = y[0][-1].item()
                if y == m * n + 1:
                    end_token_generated += 1
    
    # Compute benchmarking stats
    invalid_rate = 1
    non_pad_invalid_rate = 1
    correct_ending_rate = 0
    creativity_rate = 0
    if n_trials > 0:
        invalid_rate = invalid / n_trials
    if non_pad_invalid_possible > 0:
        non_pad_invalid_rate = non_pad_invalid / non_pad_invalid_possible
    if end_token_possible > 0:
        correct_ending_rate = end_token_generated / end_token_possible
    if original_games_possible > 0:
        creativity_rate = original_games_genrated / original_games_possible

    return (expected_perfect_loss, invalid_rate, non_pad_invalid_rate, correct_ending_rate, 
            end_token_possible, creativity_rate, original_games_possible)

                    
            # if winner == 1:
            #     counts["player_1"] += 1
            # elif winner == -1:
            #     counts["player_2"] += 1
            # elif board_full(board):
            #     counts["draw"] += 1
            # else:
            #     counts["invalid"] += 1