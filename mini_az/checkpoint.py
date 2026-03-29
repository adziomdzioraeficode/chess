"""Checkpoint save/load and RNG state management."""

import os
import random

import numpy as np
import torch

from .config import print  # flush-print
from .network import ChessNet


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _to_bytetensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.uint8)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.uint8, copy=False))
    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], int)):
        return torch.tensor(x, dtype=torch.uint8)
    return torch.as_tensor(x, dtype=torch.uint8, device="cpu")


def set_rng_state(st):
    if not st:
        return
    try:
        random.setstate(st["python"])
    except Exception:
        pass
    try:
        np.random.set_state(st["numpy"])
    except Exception:
        pass

    try:
        t = _to_bytetensor(st.get("torch"))
        if t is not None:
            torch.random.set_rng_state(t)
    except Exception as e:
        print(f"[ckpt] Warning: could not restore torch RNG: {e}")

    if torch.cuda.is_available():
        try:
            tc = st.get("torch_cuda")
            if tc is not None:
                states = [_to_bytetensor(s) for s in tc]
                states = [s for s in states if s is not None]
                if states:
                    torch.cuda.set_rng_state_all(states)
        except Exception as e:
            print(f"[ckpt] Warning: could not restore CUDA RNG: {e}")


def save_checkpoint(path: str, net: ChessNet, opt: torch.optim.Optimizer, it: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {
        "iter": it,
        "model": net.state_dict(),
        "opt": opt.state_dict(),
        "rng": get_rng_state(),
    }
    torch.save(obj, path)


def load_checkpoint(path: str, net: ChessNet,
                    opt: torch.optim.Optimizer, device: str,
                    load_opt: bool = True) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    if load_opt and ("opt" in ckpt):
        opt.load_state_dict(ckpt["opt"])
    set_rng_state(ckpt.get("rng"))
    return int(ckpt.get("iter", 0))
