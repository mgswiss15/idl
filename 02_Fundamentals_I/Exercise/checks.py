"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

checks.py: sanity checks for each exercise block.
Call check_block1(), check_block2(), etc. from the notebook after
implementing the corresponding functions in linear.py.
"""

import torch
from linear import (
    linear_scalar,
    relu_scalar,
    stack_linear,
    stack_relu,
)


def check_block1():
    """Sanity checks for linear_scalar and relu_scalar."""
    theta = torch.tensor([-1.0, 3.0])
    assert linear_scalar(2.0, theta) == 5.0,  "linear_scalar implementation not corect"
    assert relu_scalar(2.5)  == 2.5, "relu_scalar implementation not correct"
    print("Block 1 checks passed.")


def check_block2():
    """Sanity checks for stack_linear and stack_relu."""
    theta = [torch.tensor([-0.5, 2.0]), torch.tensor([1.0, -1.5]), torch.tensor([0.0, 1.0])]
    assert stack_linear(1.0, theta) == -1.25, "stack_linear implementation not correct"
    assert stack_relu(1.0, theta)   == 0.0,   "stack_relu implementation not correct"
    print("Block 2 checks passed.")