"""
Building blocks from the Lecture 15/16 language model notebook, for Homework 7 Q1.

    from model import GPTConfig, GPT
    from lm_utils import load_data, TextDataset, configure_optimizer

    train_ids, val_ids, vocab_size = load_data("data/shakespeare.txt")
    train_dataset = TextDataset(train_ids, block_size=64)
    val_dataset = TextDataset(val_ids, block_size=64)

    model_args = dict(n_layer=6, n_head=4, n_embd=256, block_size=64,
                      bias=True, vocab_size=vocab_size, dropout=0.0)
    model = GPT(GPTConfig(**model_args))
    optimizer, scheduler = configure_optimizer(model, max_iter=1000)

    logits, loss = model(X, Y)      # X, Y: a batch from a DataLoader over train_dataset
"""
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset


def load_data(path="data/shakespeare.txt"):
    """Character-level tokenization. Returns the training token ids (first 90% of the text), the validation token
    ids (last 10%) and the vocabulary size."""
    with open(path, "r") as f:
        data = f.read()
    chars = sorted(list(set(data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    n = len(data)
    train_ids = np.array([stoi[c] for c in data[:int(n * 0.9)]], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in data[int(n * 0.9):]], dtype=np.uint16)
    return train_ids, val_ids, len(chars)


class TextDataset(Dataset):
    """Sample i is the input x = tokens i .. i+block_size-1 and the target y = tokens i+1 .. i+block_size."""

    def __init__(self, text_ids, block_size):
        self.data = text_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def configure_optimizer(model, max_iter, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, weight_decay=0.1):
    """AdamW with weight decay on the 2D parameters only, and a learning rate schedule of linear warmup for
    warmup_iters iterations followed by cosine decay to min_lr at max_iter. Call scheduler.step() once per
    iteration, after optimizer.step()."""
    assert max_iter > warmup_iters, f"max_iter ({max_iter}) must be larger than warmup_iters ({warmup_iters})"
    params = [p for p in model.parameters() if p.requires_grad]
    optim_groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.99))
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=max_iter - warmup_iters, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
    return optimizer, scheduler
