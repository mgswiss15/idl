"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

linear.py: implementations of scalar and vector linear functions,
activation functions, and shallow neural networks.
"""

import torch


# ==============================================================================
# Block 1 – Scalar linear function and scalar ReLU
# ==============================================================================

def linear_scalar(x, theta):
    """Scalar linear function: f(x) = theta_1 * x + theta_0.

    Args:
        x:     scalar input (Python float)
        theta: 1-D tensor of shape (2,) — (theta_0, theta_1)

    Returns:
        Scalar output theta_1 * x + theta_0.
    """
    return theta[1] * x + theta[0]


def relu_scalar(x):
    """Scalar ReLU: relu(x) = max(0, x).

    Args:
        x: scalar input (Python float)

    Returns:
        max(0, x)
    """
    return max(0.0, x)


# ==============================================================================
# Block 2 – Shallow neural network (scalar input)
# ==============================================================================

def relu_unit(x, theta):
    """Single ReLU unit: sigma(theta_1 * x + theta_0).

    Args:
        x:     scalar input (Python float)
        theta: 1-D tensor of shape (2,) — (theta_0, theta_1)

    Returns:
        Scalar output.
    """
    return relu_scalar(linear_scalar(x, theta))


def shallow(x, theta_hidden, theta_out):
    """Shallow neural network with scalar input and scalar output.

    Computes: theta_out[0] + sum_j theta_out[j] * relu_unit(x, theta_hidden[j-1])

    Args:
        x:            scalar input (Python float)
        theta_hidden: list of k tensors of shape (2,), one per hidden unit
                      — each is (theta_j0, theta_j1)
        theta_out:    1-D tensor of shape (k+1,)
                      — theta_out[0] is bias, theta_out[j] is weight for unit j

    Returns:
        Scalar output.
    """
    out = theta_out[0]
    for j, th in enumerate(theta_hidden):
        out = out + theta_out[j + 1] * relu_unit(x, th)
    return out


# ==============================================================================
# Block 3 – Vector input, scalar output
# ==============================================================================

def linear_vector(x, theta):
    """Linear function with vector input and scalar output.

    Computes: theta_1 · x + theta_0, where theta_1 = theta[1:].

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Scalar output (0-d tensor).
    """
    return torch.dot(theta[1:], x) + theta[0]


def relu_tensor(x):
    """Element-wise ReLU for a tensor of any shape.

    Args:
        x: tensor

    Returns:
        Tensor of same shape with negative values zeroed out.
    """
    return torch.clamp(x, min=0.0)


def relu_unit_vector(x, theta):
    """Single ReLU unit with vector input: sigma(theta_1 · x + theta_0).

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1_1, ..., theta_1_d)

    Returns:
        Scalar output (0-d tensor).
    """
    return relu_tensor(linear_vector(x, theta))


def shallow_vector(x, theta_hidden, theta_out):
    """Shallow network with vector input and scalar output.

    Computes: theta_out[0] + sum_j theta_out[j] * relu_unit_vector(x, theta_hidden[j-1])

    Args:
        x:            1-D tensor, shape (d,)
        theta_hidden: list of k tensors of shape (d+1,), one per hidden unit
        theta_out:    1-D tensor of shape (k+1,)

    Returns:
        Scalar output (0-d tensor).
    """
    out = theta_out[0]
    for j, th in enumerate(theta_hidden):
        out = out + theta_out[j + 1] * relu_unit_vector(x, th)
    return out