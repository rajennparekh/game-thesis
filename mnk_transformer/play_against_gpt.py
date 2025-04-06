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
    reset_button = widgets.Button(description="Reset Game", button_style="warning")
    output = widgets.Output()

    def render_board():
        with output:
            clear_output(wait=True)
            print("🧠 vs 👤  |  Current Board\n")
            for i in range(m):
                row = []
                for j in range(n):
                    cell = board[i, j]
                    if cell == 1:
                        row.append("❌")
                    elif cell == -1:
                        row.append("⭕")
                    else:
                        row.append("⬜")
                print("      ".join(row))
            print()

    def button_click(i, j):
        nonlocal move_sequence

        if board[i, j] != 0:
            return

        board[i, j] = 1
        move_sequence.append(i * n + j)
        render_board()

        if check_winner(board, k) or is_draw(board):
            show_result()
            return

        import time
        time.sleep(0.5)  # small pause for effect

        # GPT's turn
        model_input = torch.tensor(move_sequence, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(model_input, max_new_tokens=1)
        gpt_move = out[0, -1].item()
        move_sequence.append(gpt_move)
        gi, gj = divmod(gpt_move, n)
        board[gi, gj] = -1

        render_board()

        if check_winner(board, k) or is_draw(board):
            show_result()

    def is_draw(board):
        return np.all(board != 0)

    def check_winner(board, k):
        from board_ops import check_winner as cw
        return cw(board, k)

    def show_result():
        winner = check_winner(board, k)
        if winner == 1:
            print("🎉 You win!")
        elif winner == -1:
            print("🤖 GPT wins!")
        else:
            print("🤝 It's a draw!")
        for b in board_buttons:
            for btn in b:
                btn.disabled = True

    def start_game(_):
        output.clear_output()
        first = current_player.value == "GPT goes first"
        render_board()
        if first:
            import time
            time.sleep(0.5)  # small pause for effect
            # GPT plays first move
            model_input = torch.tensor([start_token], device=device).unsqueeze(0)
            with torch.no_grad():
                out = model.generate(model_input, max_new_tokens=1)
            gpt_move = out[0, -1].item()
            move_sequence.append(gpt_move)
            gi, gj = divmod(gpt_move, n)
            board[gi, gj] = -1
            render_board()
        display_board()

    def reset_game(_):
        nonlocal board, move_sequence
        board = np.zeros((m, n), dtype=int)
        move_sequence = [start_token]
        output.clear_output()
        render_board()
        for b in board_buttons:
            for btn in b:
                btn.description = " "
                btn.disabled = False
                btn.style.button_color = None
        display_board()

    def display_board():
        board_box = widgets.VBox()
        global board_buttons
        board_buttons = []

        for i in range(m):
            row = []
            for j in range(n):
                btn = widgets.Button(description=" ", layout=widgets.Layout(width="40px"))
                btn.on_click(lambda b, i=i, j=j: button_click(i, j))
                row.append(btn)
            board_buttons.append(row)

        for i in range(m):
            for j in range(n):
                val = board[i, j]
                if val == 1:
                    board_buttons[i][j].description = "X"
                    board_buttons[i][j].disabled = True
                elif val == -1:
                    board_buttons[i][j].description = "O"
                    board_buttons[i][j].disabled = True

        board_box.children = [widgets.HBox(row) for row in board_buttons]
        display(board_box)

    start_button.on_click(start_game)
    reset_button.on_click(reset_game)
    display(widgets.VBox([current_player, widgets.HBox([start_button, reset_button]), output]))
