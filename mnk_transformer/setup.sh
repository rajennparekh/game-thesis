#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install ipykernel notebook numpy pandas matplotlib torch ipywidgets wandb

# Register kernel with Jupyter
python -m ipykernel install --user --name=ttt-env --display-name "Python (TicTacToe)"

echo "✅ Environment ready! Open your notebook and select the 'Python (TicTacToe)' kernel."