# NOT YET EDITED FOR MNK

import torch
# from tokens import START
from board_ops import check_winner, board_full, get_valid_moves
import numpy as np
from setup import load_from_checkpoint, device


model = load_from_checkpoint()
model.eval()
model.to(device)

m = model.m
n = model.n
k = model.k

start_token = m * n
with torch.no_grad():
    board = np.zeros((m, n), dtype=int)
    player = 1
    winner = None
    moves = [start_token]
    while winner is None and not board_full(board):
        if player == 1:
            x = torch.tensor(moves, dtype=torch.long, device=device)[None, ...]
            y = model.generate(x, max_new_tokens=1, temperature=1.0, top_k=3)
            y = y[0][-1].item()

            if y not in set(range(9)) or y in moves:
                print(f"AI used invalid move: {y} moves: {moves}")
                winner = None
                break

            i, j = y // 3, y % 3
        else:
            valid = [i * 3 + j for i, j in get_valid_moves(board)]
            y = None
            while y not in valid:
                y = input(f"Your move! (a number from 0-{m * n - 1}): ")
                try:
                    y = int(y)
                except:
                    print("invalid")
                    y = None

            i, j = y // 3, y % 3

        moves.append(i * 3 + j)
        board[i][j] = player

        print(board)

        player *= -1
        winner = check_winner(board, k)
