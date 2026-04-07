"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

checks.py: sanity checks for each exercise block.
Call check_block1(), check_block2(), etc. from the notebook after
implementing the corresponding functions in linear.py.
"""

import torch
from linear import linear_scalar, relu_scalar, relu_unit, shallow, linear_vector, relu_tensor


def check_block1():
    """Sanity checks for linear_scalar and relu_scalar."""
    theta = torch.tensor([-1.0, 3.0])
    assert linear_scalar(2.0, theta) == 5.0,  "linear_scalar implementation not correct"
    assert relu_scalar(2.5) == 2.5,           "relu_scalar implementation not correct"
    print("Block 1 checks passed.")


def check_block2():
    """Sanity checks for relu_unit and shallow."""
    th = torch.tensor([-0.5, 2.0])
    assert relu_unit(1.0, th)  == 1.5, "relu_unit implementation not correct"
    assert relu_unit(-1.0, th) == 0.0, "relu_unit implementation not correct"

    th1 = torch.tensor([-0.5, 2.0])
    th2 = torch.tensor([1.0, -1.5])
    theta_out = torch.tensor([0.5, 1.0, -1.0])
    assert shallow(1.0, [th1, th2], theta_out) == 2.0, "shallow implementation not correct"
    print("Block 2 checks passed.")


def check_block3():
    """Sanity checks for linear_vector and relu_tensor."""
    x = torch.tensor([1.0, 2.0])
    theta = torch.tensor([0.3, 0.5, -1.0])
    assert abs(linear_vector(x, theta).item() - (-1.2)) < 1e-5, "linear_vector implementation not correct"
    t = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.allclose(relu_tensor(t), torch.tensor([0.0, 0.0, 2.0])), "relu_tensor implementation not correct"
    print("Block 3 checks passed.")