"""
E2 – Forward Pass from Scratch: Solution Script
Introduction to Deep Learning, THWS

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
        x:       scalar input (Python float or 0-d tensor)
        theta_1: weight parameter
        theta_0: bias parameter

    Returns:
        Scalar output theta_1 * x + theta_0.
    """
    return theta_1 * x + theta_0


def relu_scalar(x):
    """Scalar ReLU: relu(x) = max(0, x).

    Args:
        x: scalar input

    Returns:
        max(0, x)
    """
    return max(0.0, x)


# --- Quick sanity checks ---
print("=" * 60)
print("BLOCK 1 – Scalar linear and ReLU")
print("=" * 60)

print(f"linear_scalar(2.0, 3.0, -1.0) = {linear_scalar(2.0, 3.0, -1.0)}")   # 5.0
print(f"linear_scalar(0.0, 3.0, -1.0) = {linear_scalar(0.0, 3.0, -1.0)}")   # -1.0
print(f"relu_scalar(2.5)  = {relu_scalar(2.5)}")    # 2.5
print(f"relu_scalar(-1.0) = {relu_scalar(-1.0)}")   # 0.0

# --- Plot: linear functions with different theta ---
x_vals = [x * 0.01 for x in range(-200, 201)]   # -2 to 2 in steps of 0.01

fig, ax = plt.subplots(figsize=(6, 4))
for theta_1, theta_0, label in [
    (1.0,  0.0, r"$\theta_1=1,\;\theta_0=0$"),
    (2.0, -1.0, r"$\theta_1=2,\;\theta_0=-1$"),
    (-1.0, 0.5, r"$\theta_1=-1,\;\theta_0=0.5$"),
]:
    y_vals = [linear_scalar(x, theta_1, theta_0) for x in x_vals]
    ax.plot(x_vals, y_vals, label=label)
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("Scalar linear functions")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/block1_linear.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {FIGURES_DIR}/block1_linear.png")


# ==============================================================================
# BLOCK 2 – Composition and stacking → piecewise linear functions
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 2 – Composition and stacking")
print("=" * 60)

# 2.1  Single ReLU unit: relu(linear(x))
def relu_unit(x, theta_1, theta_0):
    return relu_scalar(linear_scalar(x, theta_1, theta_0))

# 2.2  Two-layer composition: linear( relu( linear(x) ) )
def two_layer(x, theta_1_a, theta_0_a, theta_1_b, theta_0_b):
    hidden = relu_unit(x, theta_1_a, theta_0_a)
    return linear_scalar(hidden, theta_1_b, theta_0_b)

# 2.3  Arbitrary depth: list of (theta_1, theta_0) pairs, ReLU between layers
def stack_layers(x, params):
    """Forward pass through an arbitrary chain of linear+ReLU layers.

    The last layer has NO ReLU (linear output).

    Args:
        x:      scalar input
        params: list of (theta_1, theta_0) tuples, one per layer

    Returns:
        Scalar output after passing through all layers.
    """
    out = x
    for i, (theta_1, theta_0) in enumerate(params):
        out = linear_scalar(out, theta_1, theta_0)
        if i < len(params) - 1:          # ReLU after every layer except the last
            out = relu_scalar(out)
    return out

# --- Plot: how the function changes as we stack more layers ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

configs = [
    # (title, params list)
    ("1 ReLU unit",
     [(2.0, -0.5), (1.0, 0.0)]),
    ("2 hidden layers",
     [(2.0, -0.5), (-1.5, 1.0), (1.0, 0.0)]),
    ("4 hidden layers",
     [(2.0, -0.5), (-1.5, 1.0), (1.0, -0.5), (-0.8, 0.3), (1.0, 0.0)]),
]

for ax, (title, params) in zip(axes, configs):
    y_vals = [stack_layers(x, params) for x in x_vals]
    ax.plot(x_vals, y_vals, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")

fig.suptitle("Piecewise linear functions via stacking ReLU layers")
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/block2_stacking.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIGURES_DIR}/block2_stacking.png")


# ==============================================================================
# BLOCK 3 – Vector input, scalar output
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 3 – Vector input, scalar output")
print("=" * 60)

def linear_vector(x, theta_1, theta_0):
    """Linear function with vector input and scalar output.

    Computes the dot product theta_1 · x + theta_0.

    Args:
        x:       1-D tensor, shape (n,)
        theta_1: 1-D tensor, shape (n,)   — weight vector
        theta_0: scalar (float or 0-d tensor) — bias

    Returns:
        Scalar output (0-d tensor).
    """
    return torch.dot(theta_1, x) + theta_0


def relu_tensor(x):
    """Element-wise ReLU for a tensor."""
    return torch.clamp(x, min=0.0)


# --- Sanity check ---
x_vec = torch.tensor([1.0, 2.0])
theta_1_vec = torch.tensor([0.5, -1.0])
theta_0_scalar = 0.3
out = linear_vector(x_vec, theta_1_vec, theta_0_scalar)
print(f"linear_vector([1, 2], [0.5, -1.0], 0.3) = {out.item():.4f}")   # 0.5 - 2.0 + 0.3 = -1.2

# --- Surface plot: 2D input → scalar output ---
grid_vals = torch.linspace(-2, 2, 80)
X1, X2 = torch.meshgrid(grid_vals, grid_vals, indexing="ij")

theta_1_2d = torch.tensor([1.0, 0.5])
theta_0_2d = 0.0

# Linear surface
Z_linear = theta_1_2d[0] * X1 + theta_1_2d[1] * X2 + theta_0_2d

# ReLU surface
Z_relu = torch.clamp(Z_linear, min=0.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"})

for ax, Z, title in zip(axes,
                         [Z_linear, Z_relu],
                         ["Linear: $\\theta_1 \\cdot x + \\theta_0$",
                          "After ReLU"]):
    ax.plot_surface(X1.numpy(), X2.numpy(), Z.numpy(),
                    cmap="viridis", alpha=0.85)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    ax.set_title(title)

fig.suptitle("2D input → scalar output")
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/block3_surface.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIGURES_DIR}/block3_surface.png")


# ==============================================================================
# BLOCK 4 – Batching: loop vs. vectorised, timing comparison
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 4 – Batching: loop vs. vectorised")
print("=" * 60)

# Parameters for a linear layer: 2D input → 1D output
theta_1_batch = torch.tensor([1.0, 0.5])   # shape (2,)
theta_0_batch = torch.tensor(0.3)

# Generate N input samples
N = 100_000
X_batch = torch.randn(N, 2)   # shape (N, 2)

# --- Loop version ---
def forward_loop(X, theta_1, theta_0):
    """Compute linear output for each row of X using a Python loop."""
    outputs = []
    for i in range(X.shape[0]):
        outputs.append(torch.dot(theta_1, X[i]) + theta_0)
    return torch.stack(outputs)

t0 = time.time()
out_loop = forward_loop(X_batch, theta_1_batch, theta_0_batch)
t_loop = time.time() - t0

# --- Vectorised version ---
def forward_vectorised(X, theta_1, theta_0):
    """Compute linear output for all rows of X at once using matmul."""
    return X @ theta_1 + theta_0    # shape (N,)

t0 = time.time()
out_vec = forward_vectorised(X_batch, theta_1_batch, theta_0_batch)
t_vec = time.time() - t0

# --- Verify outputs match ---
max_diff = (out_loop - out_vec).abs().max().item()
print(f"Loop time:        {t_loop:.4f} s")
print(f"Vectorised time:  {t_vec:.4f} s")
print(f"Speed-up:         {t_loop / t_vec:.1f}x")
print(f"Max output diff:  {max_diff:.2e}   (should be ~0)")

# --- Bar chart ---
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Loop", "Vectorised"], [t_loop, t_vec], color=["steelblue", "darkorange"])
ax.set_ylabel("Wall-clock time (s)")
ax.set_title(f"Forward pass: loop vs. vectorised\n(N={N:,} samples, 2D input)")
for i, t in enumerate([t_loop, t_vec]):
    ax.text(i, t + 0.001, f"{t:.4f}s", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/block4_timing.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIGURES_DIR}/block4_timing.png")


# ==============================================================================
# BLOCK 5 – Stateful objects: LinearLayer class
# ==============================================================================
print("\n" + "=" * 60)
print("BLOCK 5 – Stateful objects: LinearLayer class")
print("=" * 60)

# The problem with functions: parameters must be passed everywhere.
# As networks grow deeper, this becomes unmanageable.
# Solution: encapsulate parameters inside an object.

class LinearLayer:
    """A single linear layer storing its own parameters.

    Parameters are stored as:
        theta_1: weight tensor, shape (out_features, in_features)
        theta_0: bias tensor,   shape (out_features,)
    """

    def __init__(self, theta_1, theta_0):
        self.theta_1 = theta_1
        self.theta_0 = theta_0

    def forward(self, x):
        """Compute theta_1 @ x + theta_0.

        Args:
            x: input tensor, shape (in_features,) or (N, in_features)

        Returns:
            Output tensor, shape (out_features,) or (N, out_features)
        """
        return x @ self.theta_1.T + self.theta_0


# --- Sanity check: compare LinearLayer with forward_vectorised ---
theta_1_layer = torch.randn(3, 2)   # 3 output features, 2 input features
theta_0_layer = torch.randn(3)
X_test = torch.randn(5, 2)          # 5 samples

layer = LinearLayer(theta_1_layer, theta_0_layer)
out_layer = layer.forward(X_test)
out_manual = X_test @ theta_1_layer.T + theta_0_layer

max_diff = (out_layer - out_manual).abs().max().item()
print(f"LinearLayer output shape: {out_layer.shape}")
print(f"Max diff vs manual:       {max_diff:.2e}   (should be ~0)")

# --- Fast finisher: MLP class built from LinearLayer objects ---
print("\n--- Fast finisher: MLP from LinearLayer objects ---")

class MLP:
    """A multi-layer perceptron built from LinearLayer objects.

    Args:
        layers: list of LinearLayer objects (in order)
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1:   # ReLU after every layer except last
                out = relu_tensor(out)
        return out


# Build a small MLP: 2 → 4 → 1
layer1 = LinearLayer(torch.randn(4, 2), torch.randn(4))
layer2 = LinearLayer(torch.randn(1, 4), torch.randn(1))
mlp = MLP([layer1, layer2])

X_mlp = torch.randn(5, 2)
out_mlp = mlp.forward(X_mlp)
print(f"MLP input shape:  {X_mlp.shape}")
print(f"MLP output shape: {out_mlp.shape}")

print("\nDone. All figures saved to the figures/ directory.")
