import torch
import os
from tokens import VOCAB_SIZE, SEQ_LENGTH
from model import GPTConfig, GPT


out_dir = "out"
seed = 1337
device  = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def init_model(n_layer=1, n_head=1, n_embed=14, dropout=0.0, bias=False, 
               attention_layer_mult=3, mlp_layer_mult=4, model_version='gpt'): 
    # these params will be passed in when function is called. Model version will let us choose which
    # type of model we want to use in the future
    print("Initializing a new model from scratch")
    config = GPTConfig(
        block_size=SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embed,
        dropout=dropout,
        bias=bias,
        attention_layer_mult=attention_layer_mult,
        mlp_layer_mult=mlp_layer_mult,
        model_version=model_version,
    )
    return GPT(config)


def load_from_checkpoint():
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(checkpoint["config"])
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": model.state_dict(),
        "config": model.config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
