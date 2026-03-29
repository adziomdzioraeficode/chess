"""Shared test fixtures."""

import pytest
import torch
import chess
import numpy as np

from mini_az.network import ChessNet
from mini_az.encoding import board_to_tensor, legal_moves_canonical


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def net():
    """A small ChessNet for fast testing."""
    return ChessNet()


@pytest.fixture
def start_board():
    return chess.Board()


@pytest.fixture
def sample_board():
    """Board after 1. e4 e5 2. Nf3."""
    b = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        b.push(chess.Move.from_uci(uci))
    return b
