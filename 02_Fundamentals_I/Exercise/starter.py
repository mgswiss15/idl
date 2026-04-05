"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

Work through each block in order. Each TODO marks something for you to implement.
Run the script after each block to check your output before moving on.

Blocks:
    1. Scalar linear function and scalar ReLU
    2. Composition and stacking → piecewise linear functions
    3. Vector input, scalar output → dot product, surface plot
    4. Batching → loop vs. vectorised forward pass, timing comparison
    5. Stateful objects → LinearLayer class
"""

import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"

# ==============================================================================
# BLOCK 1 – Scalar linear function and scalar ReLU
# ==============================================================================

def linear_scalar(x, theta_1, theta_0):
    """Scalar linear function: f(x) = theta_1 * x + theta_0.

    Args:
        x:       scalar input (Python float)
        theta_1: weight parameter
        theta_0: bias parameter

    Returns:
        Scalar output theta_1 * x + theta_0.
    """
    # TODO: implement the scalar linear function
    pass


def relu_scalar(x):
    """Scalar ReLU: relu(x) = max(0, x).

    Args:
        x: scalar input

    Returns:
        max(0, x)
    """
    # TODO: implement scalar ReLU (hint: Python's built-in max() is fine here)
    pass


# --- Quick sanity checks (do not modify) ---
print("=" * 60)
print("BLOCK 1 – Scalar linear and ReLU")
print("=" * 60)

print(f"linear_scalar(2.0, 3.0, -1.0) = {linear_scalar(2.0, 3.0, -1.0)}")  # expected: 5.0
print(f"linear_scalar(0.0, 3.0, -1.0) = {linear_scalar(0.0, 3.0, -1.0)}")  # expected: -1.0
print(f"relu_scalar(2.5)  = {relu_scalar(2.5)}")    # expected: 2.5
print(f"relu_scalar(-1.0) = {relu_scalar(-1.0)}")   # expected: 0.0

# --- TODO: Plot linear functions with different (theta_1, theta_0) pairs ---
# Use x_vals = [x * 0.01 for x in range(-200, 201)] as your x-axis.
# Plot at least three lines with different parameters.
# Label each line using its theta values.
# Save to figures/block1_linear.png.


# ==============================================================================
# BLOCK 2 – Composition and stacking → piecewise linear functions
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 2 – Composition and stacking")
print("=" * 60)

def relu_unit(x, theta_1, theta_0):
    """Apply a linear function followed by ReLU: relu(theta_1 * x + theta_0)."""
    # TODO: implement using your functions from Block 1
    pass


def stack_layers(x, params):
    """Forward pass through an arbitrary chain of linear+ReLU layers.

    The last layer has NO ReLU (linear output).

    Args:
        x:      scalar input
        params: list of (theta_1, theta_0) tuples, one per layer

    Returns:
        Scalar output after passing through all layers.

    Example:
        params = [(2.0, -0.5), (1.0, 0.0)]  →  2 layers, 1 ReLU in between
    """
    # TODO: implement the loop over params.
    # Apply linear_scalar at every layer.
    # Apply relu_scalar after every layer EXCEPT the last.
    pass


# --- TODO: Plot how the function shape changes as you add more layers ---
# Use three configs with 1, 2, and 4 hidden layers (your choice of theta values).
# Use the same x_vals as Block 1.
# Save to figures/block2_stacking.png.
#
# Question to think about: is the output always piecewise linear?
# Can you construct a set of parameters that produces a constant output?


# ==============================================================================
# BLOCK 3 – Vector input, scalar output
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 3 – Vector input, scalar output")
print("=" * 60)

def linear_vector(x, theta_1, theta_0):
    """Linear function with vector input and scalar output.

    Computes the dot product: theta_1 · x + theta_0.

    Args:
        x:       1-D tensor, shape (n,)
        theta_1: 1-D tensor, shape (n,)   — weight vector
        theta_0: scalar (float or 0-d tensor) — bias

    Returns:
        Scalar output (0-d tensor).
    """
    # TODO: implement using torch.dot
    pass


def relu_tensor(x):
    """Element-wise ReLU for a tensor."""
    # TODO: implement using torch.clamp
    pass


# --- Sanity check (do not modify) ---
x_vec = torch.tensor([1.0, 2.0])
theta_1_vec = torch.tensor([0.5, -1.0])
theta_0_scalar = 0.3
out = linear_vector(x_vec, theta_1_vec, theta_0_scalar)
print(f"linear_vector([1, 2], [0.5, -1.0], 0.3) = {out.item():.4f}")   # expected: -1.2

# --- TODO: Plot a 3D surface for 2D input → scalar output ---
# Use torch.linspace(-2, 2, 80) for both input dimensions.
# Use torch.meshgrid to construct a grid (indexing="ij").
# Plot two surfaces side by side:
#   (a) the raw linear output
#   (b) after applying relu_tensor
# Save to figures/block3_surface.png.
#
# Question to think about: what does the ReLU surface look like geometrically?
# Can this single ReLU unit fit an arbitrary function over 2D inputs?


# ==============================================================================
# BLOCK 4 – Batching: loop vs. vectorised
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 4 – Batching: loop vs. vectorised")
print("=" * 60)

# Parameters for a linear layer: 2D input → 1D output
theta_1_batch = torch.tensor([1.0, 0.5])
theta_0_batch = torch.tensor(0.3)

# N input samples, each of dimension 2
N = 100_000
X_batch = torch.randn(N, 2)   # shape (N, 2)


def forward_loop(X, theta_1, theta_0):
    """Compute linear output for each row of X using a Python loop.

    Args:
        X:       tensor, shape (N, 2)
        theta_1: weight tensor, shape (2,)
        theta_0: scalar bias

    Returns:
        Output tensor, shape (N,)
    """
    # TODO: loop over X.shape[0], compute torch.dot(theta_1, X[i]) + theta_0
    # for each row, collect results in a list, return torch.stack(outputs)
    pass


def forward_vectorised(X, theta_1, theta_0):
    """Compute linear output for all rows of X at once.

    Args:
        X:       tensor, shape (N, 2)
        theta_1: weight tensor, shape (2,)
        theta_0: scalar bias

    Returns:
        Output tensor, shape (N,)
    """
    # TODO: implement in one line using the @ operator
    # Hint: X @ theta_1 gives a dot product for each row simultaneously
    pass


# --- Timing comparison (do not modify) ---
t0 = time.time()
out_loop = forward_loop(X_batch, theta_1_batch, theta_0_batch)
t_loop = time.time() - t0

t0 = time.time()
out_vec = forward_vectorised(X_batch, theta_1_batch, theta_0_batch)
t_vec = time.time() - t0

max_diff = (out_loop - out_vec).abs().max().item()
print(f"Loop time:        {t_loop:.4f} s")
print(f"Vectorised time:  {t_vec:.4f} s")
print(f"Speed-up:         {t_loop / t_vec:.1f}x")
print(f"Max output diff:  {max_diff:.2e}   (should be ~0)")

# --- TODO: Make a bar chart comparing loop vs. vectorised wall-clock time ---
# Save to figures/block4_timing.png.


# ==============================================================================
# BLOCK 5 – Stateful objects: LinearLayer class
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 5 – Stateful objects: LinearLayer class")
print("=" * 60)

# Problem: so far, parameters (theta_1, theta_0) are loose variables that
# must be passed into every function call. As networks grow deeper, this
# becomes unwieldy. A better approach: encapsulate parameters inside an object.

class LinearLayer:
    """A single linear layer that stores its own parameters.

    Parameters:
        theta_1: weight tensor, shape (out_features, in_features)
        theta_0: bias tensor,   shape (out_features,)
    """

    def __init__(self, theta_1, theta_0):
        # TODO: store theta_1 and theta_0 as instance attributes
        pass

    def forward(self, x):
        """Compute theta_1 @ x + theta_0.

        Args:
            x: input tensor, shape (in_features,) or (N, in_features)

        Returns:
            Output tensor, shape (out_features,) or (N, out_features)
        """
        # TODO: implement the linear transformation
        # Hint: for batched input use x @ self.theta_1.T + self.theta_0
        pass


# --- Sanity check (do not modify) ---
theta_1_layer = torch.randn(3, 2)
theta_0_layer = torch.randn(3)
X_test = torch.randn(5, 2)

layer = LinearLayer(theta_1_layer, theta_0_layer)
out_layer = layer.forward(X_test)
out_manual = X_test @ theta_1_layer.T + theta_0_layer

max_diff = (out_layer - out_manual).abs().max().item()
print(f"LinearLayer output shape: {out_layer.shape}")           # expected: torch.Size([5, 3])
print(f"Max diff vs manual:       {max_diff:.2e}   (should be ~0)")


# ==============================================================================
# FAST FINISHER – MLP class built from LinearLayer objects
# ==============================================================================
print("\n--- Fast finisher: MLP from LinearLayer objects ---")

class MLP:
    """A multi-layer perceptron built from a list of LinearLayer objects.

    ReLU is applied after every layer except the last.

    Args:
        layers: list of LinearLayer objects (in order, input → output)
    """

    def __init__(self, layers):
        # TODO: store the list of layers
        pass

    def forward(self, x):
        # TODO: pass x through each layer in order,
        # applying relu_tensor after every layer except the last
        pass


# --- Sanity check (do not modify) ---
layer1 = LinearLayer(torch.randn(4, 2), torch.randn(4))
layer2 = LinearLayer(torch.randn(1, 4), torch.randn(1))
mlp = MLP([layer1, layer2])

X_mlp = torch.randn(5, 2)
out_mlp = mlp.forward(X_mlp)
print(f"MLP input shape:  {X_mlp.shape}")    # expected: torch.Size([5, 2])
print(f"MLP output shape: {out_mlp.shape}")  # expected: torch.Size([5, 1])

print("\nDone. Check the figures/ directory for your plots.")
