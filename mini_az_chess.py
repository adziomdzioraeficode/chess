#!/usr/bin/env python3
"""
mini_az_chess.py  --  thin wrapper that delegates to the mini_az package.

The original monolithic source is preserved as mini_az_chess.py.bak.
"""
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from mini_az.main import main  # noqa: E402

if __name__ == "__main__":
    main()
