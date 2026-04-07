"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

linear.py: implementations of scalar and vector linear functions,
activation functions, batched forward passes, and stateful layer classes.
"""

import torch


# ==============================================================================
# Scalar functions
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


def stack_linear(x, theta):
    """Composition of multiple scalar linear functions.

    Applies k linear functions in sequence with no activation in between.
    Note: a composition of linear functions is itself linear.

    Args:
        x:     scalar input (Python float)
        theta: list of 1-D tensors of shape (2,), one per layer — (theta_0, theta_1)

    Returns:
        Scalar output.
    """
    out = x
    for th in theta:
        out = linear_scalar(out, th)
    return out


def stack_relu(x, theta):
    """Composition of multiple scalar linear functions with ReLU activations.

    Applies ReLU after every layer except the last.

    Args:
        x:     scalar input (Python float)
        theta: list of 1-D tensors of shape (2,), one per layer — (theta_0, theta_1)

    Returns:
        Scalar output.
    """
    out = x
    for i, th in enumerate(theta):
        out = linear_scalar(out, th)
        if i < len(theta) - 1:
            out = relu_scalar(out)
        # out = relu_scalar(out)
    return out


# ==============================================================================
# Vector functions
# ==============================================================================

def linear_vector(x, theta):
    """Linear function with vector input and scalar output.

    Computes the dot product: theta_1 · x + theta_0.

    Args:
        x:     1-D tensor, shape (d,)
        theta: 1-D tensor, shape (d+1,) — (theta_0, theta_1, ..., theta_d)

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


# ==============================================================================
# Batched forward passes
# ==============================================================================

def forward_loop(X, theta_1, theta_0):
    """Linear forward pass over a batch using a Python loop.

    Args:
        X:       tensor, shape (N, n)
        theta_1: weight tensor, shape (n,)
        theta_0: scalar bias

    Returns:
        Output tensor, shape (N,)
    """
    outputs = []
    for i in range(X.shape[0]):
        outputs.append(torch.dot(theta_1, X[i]) + theta_0)
    return torch.stack(outputs)


def forward_vectorised(X, theta_1, theta_0):
    """Linear forward pass over a batch using matrix multiplication.

    Args:
        X:       tensor, shape (N, n)
        theta_1: weight tensor, shape (n,)
        theta_0: scalar bias

    Returns:
        Output tensor, shape (N,)
    """
    return X @ theta_1 + theta_0


# ==============================================================================
# Stateful layer classes
# ==============================================================================

class LinearLayer:
    """A single linear layer that stores its own parameters.

    Parameters theta_1 (weights) and theta_0 (bias) are stored as instance
    attributes so they travel with the layer rather than being passed around
    as loose variables.

    Args:
        theta_1: weight tensor, shape (out_features, in_features)
        theta_0: bias tensor,   shape (out_features,)
    """

    def __init__(self, theta_1, theta_0):
        self.theta_1 = theta_1
        self.theta_0 = theta_0

    def forward(self, x):
        """Compute theta_1 @ x + theta_0.

        Args:
            x: tensor, shape (in_features,) or (N, in_features)

        Returns:
            Tensor, shape (out_features,) or (N, out_features)
        """
        return x @ self.theta_1.T + self.theta_0


class MLP:
    """A multi-layer perceptron built from a list of LinearLayer objects.

    ReLU is applied after every layer except the last (linear output).

    Args:
        layers: list of LinearLayer objects, ordered from input to output
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """Forward pass through all layers.

        Args:
            x: input tensor, shape (in_features,) or (N, in_features)

        Returns:
            Output tensor.
        """
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1:
                out = relu_tensor(out)
        return out
