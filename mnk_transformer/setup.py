import torch
import os
# from tokens import VOCAB_SIZE, SEQ_LENGTH
from model import GPTConfig, GPT


out_dir = "out"
seed = 1337
device  = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def init_model(m, n, k, n_layer=1, n_head=1, n_embed=12, dropout=0.0, bias=False, 
               mlp_layer_mult=4): 
    # these params will be passed in when function is called. Model version will let us choose which
    # type of model we want to use in the future
    print("Initializing a new model from scratch")
    config = GPTConfig(
        m=m,
        n=n,
        k=k,
        # block_size=m * n + 1, #SEQ_LENGTH,
        # vocab_size=m * n + 2, #VOCAB_SIZE,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embed,
        dropout=dropout,
        bias=bias,
        mlp_layer_mult=mlp_layer_mult,
    )
    return GPT(config)


def load_from_checkpoint():
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(checkpoint["config"])
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model, m=None, n=None, k=None):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": model.state_dict(),
        "config": model.config,
    }
    print(f"saving checkpoint to {out_dir}")
    if m is None or n is None or k is None: 
        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    else:
        torch.save(checkpoint, os.path.join(out_dir, f"ckpt_m{m}_n{n}_k{k}.pt"))

