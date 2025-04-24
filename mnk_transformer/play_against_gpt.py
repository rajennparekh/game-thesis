import torch
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from setup import device

def play_game(model, m=3, n=3, k=3):
    model.eval()
    model.to(device)
    pad_token = m * n + 1
    start_token = m * n

    board = np.zeros((m, n), dtype=int)
    move_sequence = [start_token]

    current_player = widgets.ToggleButtons(
        options=["You go first", "GPT goes first"],
        description="Who starts:",
        style={"description_width": "initial"},
    )
    start_button = widgets.Button(description="Start Game", button_style="success")
    output = widgets.Output()

    board_buttons = [[None for _ in range(n)] for _ in range(m)]  # will get filled in

    def update_board_buttons():
        for i in range(m):
            for j in range(n):
                val = board[i, j]
                btn = board_buttons[i][j]
                if val == 1:
                    btn.description = "❌"
                    btn.disabled = True
                elif val == -1:
                    btn.description = "⭕"
                    btn.disabled = True
                else:
                    btn.description = "⬜"

    def is_draw(board):
        return np.all(board != 0)

    def check_winner(board, k):
        from board_ops import check_winner as cw
        return cw(board, k)

    def show_result():
        winner = check_winner(board, k)
        if winner == 1:
            output.clear_output()
            with output:
                print("🎉 You win!")
        elif winner == -1:
            output.clear_output()
            with output:
                print("🤖 GPT wins!")
        else:
            output.clear_output()
            with output:
                print("🤝 It's a draw!")

        for row in board_buttons:
            for btn in row:
                btn.disabled = True

    def button_click(i, j):
        nonlocal move_sequence

        if board[i, j] != 0:
            return

        # Human move
        board[i, j] = 1
        move_sequence.append(i * n + j)
        update_board_buttons()

        if check_winner(board, k) or is_draw(board):
            show_result()
            return

        # GPT move
        import time
        time.sleep(0.5)

        model_input = torch.tensor(move_sequence, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(model_input, max_new_tokens=1)
        gpt_move = out[0, -1].item()
        move_sequence.append(gpt_move)
        gi, gj = divmod(gpt_move, n)
        board[gi, gj] = -1
        update_board_buttons()

        if check_winner(board, k) or is_draw(board):
            show_result()

    def display_board():
        board_box = widgets.VBox()

        for i in range(m):
            row = []
            for j in range(n):
                btn = widgets.Button(
                    description="⬜",
                    layout=widgets.Layout(width="120px", height="120px"),
                    style={'font_size': '36px'}
                )
                btn.on_click(lambda b, i=i, j=j: button_click(i, j))
                board_buttons[i][j] = btn
                row.append(btn)
            board_box.children += (widgets.HBox(row),)

        display(board_box)

    def start_game(_):
        board[:, :] = 0
        move_sequence.clear()
        move_sequence.append(start_token)
        output.clear_output()

        for row in board_buttons:
            for btn in row:
                btn.description = "⬜"
                btn.disabled = False

        first = current_player.value == "GPT goes first"
        if first:
            import time
            time.sleep(0.5)
            model_input = torch.tensor([start_token], device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(model_input, max_new_tokens=1)
            gpt_move = out[0, -1].item()
            move_sequence.append(gpt_move)
            gi, gj = divmod(gpt_move, n)
            board[gi, gj] = -1

        update_board_buttons()

    start_button.on_click(start_game)
    display(widgets.VBox([current_player, widgets.HBox([start_button]), output]))
    display_board()
