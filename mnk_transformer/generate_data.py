import numpy as np
import copy
from collections import defaultdict
import argparse
import random
# from tokens import PAD, SEQ_LENGTH, START
from board_ops import check_winner, board_full, optimal_moves, get_valid_moves

def sample_trajectories(board, k, num_samples=1_000_000):
    print("Board large, Running sampling")
    m, n = board.shape
    trajectories = []
    
    for x in range(num_samples):
        if x % 1000 == 0:
            print(x)
        current_board = np.zeros((m, n), dtype=int)
        current_seq = []
        current_player = 1  # Assume player 1 starts
        moves = [(i, j) for i in range(m) for j in range(n)]  # List of available moves
        
        while True:
            # Check terminal state
            winner = check_winner(current_board, k)
            if winner is not None or not moves:
                trajectories.append((current_board.copy(), current_seq.copy(), winner))
                break
            
            # Choose a random move
            i, j = random.choice(moves)
            current_board[i][j] = current_player
            current_seq.append(i * n + j)
            moves.remove((i, j))  # Remove the chosen move
            current_player = -current_player  # Switch player

    return trajectories

def all_trajectories(board, seq, player, k):
    from collections import deque
    
    m, n = board.shape
    stack = deque([(board.copy(), seq.copy(), player)])
    trajectories = []

    while stack:
        current_board, current_seq, current_player = stack.pop()

        # Check for terminal state
        winner = check_winner(current_board, k)
        if winner is not None or board_full(current_board):
            trajectories.append((current_board.copy(), current_seq.copy(), winner))
            continue

        # Explore possible moves
        for i in range(m):
            for j in range(n):
                if current_board[i][j] == 0:
                    current_board[i][j] = current_player
                    current_seq.append(i * n + j)
                    stack.append((current_board.copy(), current_seq.copy(), -current_player))
                    current_board[i][j] = 0
                    current_seq.pop()
    
    return trajectories

def seq_to_board(seq, m, n):
    board = np.zeros((m, n), dtype=int)
    player = 1
    for s in seq:
        i, j = s // n, s % n
        board[i][j] = player
        player = -player
    return board


def save_data(trajectories, m, n, k):
    outcomes = defaultdict(int)
    for b, s, w in trajectories:
        outcomes[w] += 1

    print(outcomes)

    data = np.full((len(trajectories), m * n + 1), m * n + 1, dtype=np.int16)
    for i in range(len(trajectories)):
        row = [m * n] + trajectories[i][1]
        if i < 10:
            print(row)
        data[i, : len(row)] = row

    np.random.shuffle(data)

    filename = f"data/train_m{m}_n{n}_k{k}.npy"

    np.save(filename, data)

def run_generation(m, n, k):
    board = np.zeros((m, n), dtype=int)
    # if m >= 4 or n >= 4 or k >= 4:
    #     trajectories = sample_trajectories(board, k)
    # else:
    #     trajectories = all_trajectories(board, [], 1, k)
        
    n_trajectories = 100000 if m * n < 10 else 1000000
    trajectories = sample_trajectories(board, k, n_trajectories)
    save_data(trajectories, m, n, k)


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate and save trajectories for a given m x n board.")
    parser.add_argument('--m', type=int, default=3, help="Number of rows in the board (default: 3).")
    parser.add_argument('--n', type=int, default=3, help="Number of columns in the board (default: 3).")
    parser.add_argument('--k', type=int, default=3, help="Parameter k for trajectory generation (default: 3).")
    # Parse the arguments
    args = parser.parse_args()

    # Board setup and trajectory generation
    m, n, k = args.m, args.n, args.k
    trajectories = run_generation(m, n, k)