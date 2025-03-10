import torch
from tokens import PAD
## PAD should be m x n + 1
import numpy as np

#where do we use these functions??
def winning_moves(board, player, k):
    m, n = board.shape
    moves = []
    for i in range(m):
        for j in range(n):
            if board[i][j] == 0:
                board[i][j] = player
                if check_winner(board, k) == player:
                    moves.append((i, j))
                board[i][j] = 0
    return moves


def board_full(board):
    return 0 not in board


def check_winner(board, k):
    """
    Check if there are k squares in a row on an m x n tic-tac-toe board.

    Args:
        board (np.ndarray): A 2D numpy array representing the board.
        k (int): The number of squares in a row needed for a win.

    Returns:
        int or None: The player with k in a row (-1 or 1), or None if no winner.
    """
    m, n = board.shape

    for player in [-1, 1]:
        # Check rows
        for i in range(m):
            for j in range(n - k + 1):
                if all(board[i, j:j + k] == player):
                    return player

        # Check columns
        for j in range(n):
            for i in range(m - k + 1):
                if all(board[i:i + k, j] == player):
                    return player

        # Check diagonals (\ direction)
        for i in range(m - k + 1):
            for j in range(n - k + 1):
                if all(board[i + d, j + d] == player for d in range(k)):
                    return player

        # Check diagonals (/ direction)
        for i in range(m - k + 1):
            for j in range(k - 1, n):
                if all(board[i + d, j - d] == player for d in range(k)):
                    return player

    return None


import numpy as np

def optimal_moves(board, player, k): # THIS FUNCTION IS A WORK IN PROGRESS!!
    """
    Decide the next move for the given player on an m x n board.

    Args:
        board (np.ndarray): The current state of the board (2D numpy array).
        player (int): The player making the move (1 or -1).
        k (int): The number of squares in a row needed for a win.

    Returns:
        tuple: The row and column of the chosen move.
    """
    m, n = board.shape
    opponent = -player

    # 1. Check for a winning move for the current player
    wins = winning_moves(board, player, k)
    if len(wins) > 0:
        return wins[0]

    # 2. Block the opponent if they can win
    blocks = winning_moves(board, opponent, k)
    if len(blocks) > 0:
        return blocks[0]

    # 3. Play in or near the center of the board
    center = (m // 2, n // 2)
    if board[center] == 0:
        return center
    else:
        # Prioritize moves near the center
        moves = sorted(
            [(i, j) for i in range(m) for j in range(n) if board[i, j] == 0],
            key=lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1]),
        )
        return moves[0]

    # 4. Extend unblocked stretches (prioritized implicitly by center-first)
    # If no better strategy applies, just take the first available move
    for i in range(m):
        for j in range(n):
            if board[i, j] == 0:
                return (i, j)



def get_valid_moves(board):
    m, n = board.shape
    return [(i, j) for i in range(m) for j in range(n) if board[i, j] == 0]


def batch_board_full(boards):
    return ~(boards == 0).any(dim=2).any(dim=1)


def batch_check_winner(boards, k):
    """
    Check if there are k squares in a row on m x n tic-tac-toe boards for a batch.

    Args:
        boards (torch.Tensor): A 3D tensor of shape (batch_size, m, n) representing the boards.
        k (int): The number of squares in a row needed for a win.

    Returns:
        torch.Tensor: A 1D tensor of winners for each board (-1, 1, or 0 if no winner).
    """
    batch_size, m, n = boards.shape
    players = torch.tensor([-1, 1], device=boards.device)

    def check_line_sums(tensor, dim):
        """Helper function to calculate k-in-a-row sums along a specific dimension."""
        shape = tensor.shape
        unfold = tensor.unfold(dim, k, 1)
        return unfold.sum(dim=-1).view(*shape[:-1], -1)

    # Check rows and columns
    row_sums = check_line_sums(boards, 2)  # (batch_size, m, n-k+1)
    col_sums = check_line_sums(boards.transpose(1, 2), 2)  # (batch_size, n, m-k+1)

    # Check diagonals (\\ direction)
    diag_sums = []
    for offset in range(-(m - k), n - k + 1):
        diag = boards.diagonal(offset=offset, dim1=1, dim2=2)  # Extract diagonals
        if diag.size(-1) >= k:
            diag_sums.append(check_line_sums(diag, 1))
    diag_sums = torch.cat(diag_sums, dim=1) if diag_sums else torch.zeros((batch_size, 0), device=boards.device)

    # Check anti-diagonals (/ direction)
    flipped_boards = boards.flip(dims=(2,))
    anti_diag_sums = []
    for offset in range(-(m - k), n - k + 1):
        anti_diag = flipped_boards.diagonal(offset=offset, dim1=1, dim2=2)
        if anti_diag.size(-1) >= k:
            anti_diag_sums.append(check_line_sums(anti_diag, 1))
    anti_diag_sums = torch.cat(anti_diag_sums, dim=1) if anti_diag_sums else torch.zeros((batch_size, 0), device=boards.device)

    # Concatenate all sums (ensure same number of dimensions)
    all_sums = torch.cat(
        [row_sums.flatten(1), col_sums.flatten(1), diag_sums, anti_diag_sums], dim=1
    )

    # Check for wins
    matches = (all_sums.unsqueeze(-1) == (k * players)).any(dim=1)

    # Determine winners
    winners = torch.where(
        matches[:, 1], players[1], torch.where(matches[:, 0], players[0], 0)
    )

    return winners


def batch_detect_illegal_moves(batch_seq, m, n, ):#pad_token=PAD,):
    # CHECK THAT THIS IS DOING THE RIGHT THING. CAN WE SAY PAD = SEQ_LEN? 
    pad_token = m * n + 1
    batch_size, seq_len = batch_seq.shape

    # Create masks for valid moves (0-m*n - 1) and pad tokens
    valid_moves = (batch_seq >= 0) & (batch_seq < m * n)
    pad_mask = batch_seq == pad_token

    # Ensure all tokens in sequences are either valid moves or PAD
    valid_seq = valid_moves | pad_mask

    # Create safe indices to avoid out-of-bounds during scatter
    safe_batch_seq = torch.where(valid_moves, batch_seq, torch.zeros_like(batch_seq))

    # Convert sequences to one-hot encoding to detect repeated moves
    one_hot_moves = torch.zeros(
        batch_size, seq_len, m * n, dtype=torch.float, device=batch_seq.device
    )
    one_hot_moves.scatter_(
        2, safe_batch_seq.unsqueeze(-1), valid_moves.float().unsqueeze(-1)
    )

    # Sum along the sequence length dimension to count occurrences of each move
    move_counts = one_hot_moves.sum(dim=1)

    # Moves that are repeated (count > 1) are illegal
    repeated_moves = move_counts > 1

    # Also, ensure all tokens before PAD tokens in a sequence are valid moves
    no_invalid_before_pad = ~pad_mask | valid_moves.cumsum(dim=1).bool()

    # Final validity check for each sequence in the batch
    valid_sequences = (
        valid_seq.all(dim=1)
        & ~repeated_moves.any(dim=1)
        & no_invalid_before_pad.all(dim=1)
    )

    return ~valid_sequences


def batch_seq_to_board(batch_seq, m, n,):# pad_token=PAD):
    # CHECK THIS. DOES PAD = SEQ_LEN?
    pad_token = m * n + 1
    batch_size, seq_len = batch_seq.shape
    one_hot = torch.zeros(batch_size, seq_len, m * n, device=batch_seq.device)

    # Mask to identify non-padding tokens
    valid_mask = batch_seq != pad_token

    # Mask to identify valid move tokens
    move_mask = (batch_seq >= 0) & (batch_seq <= m * n)

    # Combine masks to identify valid non-padding move tokens
    combined_mask = valid_mask & move_mask

    # Adjust the sequence length based on non-padding tokens
    valid_lengths = combined_mask.sum(dim=1)

    # Create one-hot encoded tensor for each valid sequence entry in the batch
    br = torch.arange(batch_size, device=batch_seq.device).unsqueeze(1)
    sr = torch.arange(seq_len, device=batch_seq.device)

    safe_batch_seq = torch.where(combined_mask, batch_seq, torch.zeros_like(batch_seq))

    one_hot[
        br,
        sr,
        safe_batch_seq,
    ] = combined_mask.float()

    # Generate player matrix [-1, 1, -1, 1, ...]
    max_players = torch.tensor([1, -1] * (seq_len // 2 + 1), device=batch_seq.device)
    players = [max_players[:seq_len].view(seq_len, 1) for _ in valid_lengths]
    players_padded = torch.nn.utils.rnn.pad_sequence(players, batch_first=True)

    # Matrix multiplication
    one_hot = one_hot.float()
    players_padded = players_padded.float()
    batch_board = torch.bmm(one_hot.transpose(1, 2), players_padded).squeeze(2)

    return batch_board.view(batch_size, m, n)


def batch_next_valid_move_from_seq(batch_seq, m, n):
    all_moves = torch.arange(m * n, device=batch_seq.device)
    available_moves = [set(all_moves.tolist()) - set(seq.tolist()) for seq in batch_seq]
    next_moves = [
        list(moves)[torch.randint(0, len(moves), (1,)).item()] if moves else -1
        for moves in available_moves
    ]
    return torch.tensor(next_moves, device=batch_seq.device)
