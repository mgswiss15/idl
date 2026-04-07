"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

checks.py: sanity checks for each exercise block.
Call check_block1(), check_block2(), etc. from the notebook after
implementing the corresponding functions in linear.py.
"""

import torch
from linear import *


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


def check_block3():
    """Sanity checks for linear_vector and relu_tensor."""
    x = torch.tensor([1.0, 2.0])
    theta = torch.tensor([0.3, 0.5, -1.0])   # theta_0=0.3, theta_1=[0.5, -1.0]
    # 0.5*1 + (-1.0)*2 + 0.3 = -1.2
    assert abs(linear_vector(x, theta).item() - (-1.2)) < 1e-5, "linear_vector implementation not correct"
    t = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.allclose(relu_tensor(t), torch.tensor([0.0, 0.0, 2.0])), "relu_tensor implementation not correct"
    print("Block 3 checks passed.")