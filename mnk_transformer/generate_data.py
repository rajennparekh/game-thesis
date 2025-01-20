import numpy as np
import copy
from collections import defaultdict
import argparse

# from tokens import PAD, SEQ_LENGTH, START
from board_ops import check_winner, board_full, optimal_moves, get_valid_moves


def all_trajectories(board, seq, player, k):
    m, n = board.shape
    winner = check_winner(board, k)
    if winner is not None or board_full(board):
        return [(board.copy(), copy.copy(seq), winner)]
    trajectories = []
    for i in range(m):
        for j in range(n):
            if board[i][j] == 0:
                board[i][j] = player
                seq.append((i * n + j))
                trajectories += all_trajectories(board, seq, -player, k)
                board[i][j] = 0
                seq.pop()
    return trajectories


def all_optimal_trajectories(board, seq, player, k, optimal_player={1}): #test this, more complicated
    m, n = board.shape
    winner = check_winner(board, k)
    if winner is not None or board_full(board):
        return [(board.copy(), copy.copy(seq), winner)]
    trajectories = []
    moves = (
        optimal_moves(board, player, k)
        if player in optimal_player
        else get_valid_moves(board)
    )
    if not moves:
        moves = get_valid_moves(board)
    for i, j in moves:
        board[i][j] = player
        seq.append((i * n + j))
        trajectories += all_optimal_trajectories(board, seq, -player, k)
        board[i][j] = 0
        seq.pop()
    return trajectories


def seq_to_board(seq, m, n):
    board = np.zeros((m, n), dtype=int)
    player = 1
    for s in seq:
        i, j = s // n, s % n
        board[i][j] = player
        player = -player
    return board


def save_data(trajectories, m, n, k, all_or_optimal):
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

    filename = f"data/train_m{m}_n{n}_k{k}_{all_or_optimal}.npy"

    np.save(filename, data)


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate and save trajectories for a given m x n board.")
    parser.add_argument('--m', type=int, default=3, help="Number of rows in the board (default: 3).")
    parser.add_argument('--n', type=int, default=3, help="Number of columns in the board (default: 3).")
    parser.add_argument('--k', type=int, default=3, help="Parameter k for trajectory generation (default: 3).")
    parser.add_argument('--all_or_optimal', type=str, default='all', help="Type 'all' or 'optimal', default to all")
    # Parse the arguments
    args = parser.parse_args()

    # Board setup and trajectory generation
    m, n, k, all_or_optimal = args.m, args.n, args.k, args.all_or_optimal
    board = np.zeros((m, n), dtype=int)
    trajectories = all_trajectories(board, [], 1, k)
    # trajectories = all_optimal_trajectories(board, [], 1, {-1, 1})
    # trajectories = all_optimal_trajectories(board, [], 1)
    
    # Output and save data
    print(len(trajectories))
    save_data(trajectories, m, n, k, all_or_optimal)
