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

def linear_scalar(x, theta_1, theta_0):
    """Scalar linear function: f(x) = theta_1 * x + theta_0.

    Args:
        x:       scalar input (Python float)
        theta_1: weight parameter (Python float)
        theta_0: bias parameter (Python float)

    Returns:
        Scalar output theta_1 * x + theta_0.
    """
    return theta_1 * x + theta_0


def relu_scalar(x):
    """Scalar ReLU: relu(x) = max(0, x).

    Args:
        x: scalar input (Python float)

    Returns:
        max(0, x)
    """
    return max(0.0, x)


def relu_unit(x, theta_1, theta_0):
    """Single ReLU unit: relu(theta_1 * x + theta_0).

    Args:
        x:       scalar input
        theta_1: weight parameter
        theta_0: bias parameter

    Returns:
        Scalar output.
    """
    return relu_scalar(linear_scalar(x, theta_1, theta_0))


def stack_layers(x, params):
    """Forward pass through an arbitrary chain of scalar linear+ReLU layers.

    ReLU is applied after every layer except the last.

    Args:
        x:      scalar input (Python float)
        params: list of (theta_1, theta_0) tuples, one per layer

    Returns:
        Scalar output after passing through all layers.

    Example:
        stack_layers(1.0, [(2.0, -0.5), (1.0, 0.0)])
        → relu(2.0 * 1.0 - 0.5) = relu(1.5) = 1.5
        → 1.0 * 1.5 + 0.0 = 1.5
    """
    out = x
    for i, (theta_1, theta_0) in enumerate(params):
        out = linear_scalar(out, theta_1, theta_0)
        if i < len(params) - 1:
            out = relu_scalar(out)
    return out


# ==============================================================================
# Vector functions
# ==============================================================================

def linear_vector(x, theta_1, theta_0):
    """Linear function with vector input and scalar output.

    Computes the dot product: theta_1 · x + theta_0.

    Args:
        x:       1-D tensor, shape (n,)
        theta_1: 1-D tensor, shape (n,)
        theta_0: scalar (float or 0-d tensor)

    Returns:
        Scalar output (0-d tensor).
    """
    return torch.dot(theta_1, x) + theta_0


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
