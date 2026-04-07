"""
E2 – Forward Pass from Scratch
Introduction to Deep Learning, THWS

plotting.py: visualisations for each exercise block.
Each function returns a matplotlib Figure and optionally saves or displays it.

Usage from notebook:
    import plotting

    fig = plotting.plot_func(func=linear_scalar, x_vals=x_scalar, theta=theta)

    fig, ax = plt.subplots()
    plotting.plot_func(func=linear_scalar, x_vals=x_scalar, theta=theta1, ax=ax)
    plotting.plot_func(func=stack_linear,  x_vals=x_scalar, theta=thetas, ax=ax,
                       label="stack_linear", save_path='figures/stack_linear')
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_func(func, x_vals, theta=None, label=None, title=None, ax=None, show=False, save_path=None):
    """Plot a scalar function over a range of x values.

    Can draw onto an existing Axes (for overlaying multiple functions) or
    create a new Figure if none is provided.

    Args:
        func:      callable — either f(x) or f(x, theta)
        x_vals:    list of scalar x values
        theta:     tensor or list of tensors for parametric functions, or None
        label:     legend label — if None, a default is generated from theta or func name
        title:     plot title — if None, a default is generated from theta or func name
        ax:        existing matplotlib Axes to draw onto — if None, a new figure is created
        show:      if True, display the plot interactively
        save_path: if given, save to this path (.png appended if missing)

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    if theta is not None:
        y_vals = [func(x, theta) for x in x_vals]
        if label is None:
            if isinstance(theta, list):
                label = func.__name__
            else:
                label = rf"$\theta = ({theta[0].item():.2f}, {theta[1].item():.2f})$"
    else:
        y_vals = [func(x) for x in x_vals]
        if label is None:
            label = func.__name__

    if title is None:
        title = func.__name__


    ax.plot(x_vals, y_vals, label=label)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        if not save_path.endswith(".png"):
            save_path = save_path + ".png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig



# ==============================================================================
# Block 1 – Scalar linear functions
# ==============================================================================

def plot_linear_scalar(show=False, save_path=None):
    """Plot scalar linear functions for several (theta_1, theta_0) pairs.

    Shows that f(x) = theta_1 * x + theta_0 is always a straight line,
    parameterised by slope and intercept.
    """
    params = [
        (1.0,  0.0, r"$\theta_1=1,\;\theta_0=0$"),
        (2.0, -1.0, r"$\theta_1=2,\;\theta_0=-1$"),
        (-1.0, 0.5, r"$\theta_1=-1,\;\theta_0=0.5$"),
        (0.5,  1.0, r"$\theta_1=0.5,\;\theta_0=1$"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    for theta_1, theta_0, label in params:
        y = [linear_scalar(x, theta_1, theta_0) for x in _X_SCALAR]
        ax.plot(_X_SCALAR, y, label=label)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Scalar linear functions")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return _maybe_save_show(fig, save_path, show)


def plot_relu_scalar(show=False, save_path=None):
    """Plot ReLU applied to a linear function, illustrating the hinge."""
    theta_1, theta_0 = 1.5, -0.5

    y_linear = [linear_scalar(x, theta_1, theta_0) for x in _X_SCALAR]
    y_relu   = [relu_scalar(linear_scalar(x, theta_1, theta_0)) for x in _X_SCALAR]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(_X_SCALAR, y_linear, color="steelblue")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_title(r"Linear: $\theta_1 x + \theta_0$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$f(x)$")

    axes[1].plot(_X_SCALAR, y_relu, color="darkorange")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_title(r"After ReLU: $\max(0,\,\theta_1 x + \theta_0)$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$f(x)$")

    fig.suptitle("The ReLU hinge")
    fig.tight_layout()

    return _maybe_save_show(fig, save_path, show)


# ==============================================================================
# Block 2 – Stacking layers → piecewise linear functions
# ==============================================================================

def plot_stacking(show=False, save_path=None):
    """Plot how the function shape changes as more ReLU layers are stacked."""
    configs = [
        ("1 ReLU unit",
         [(2.0, -0.5), (1.0, 0.0)]),
        ("2 hidden layers",
         [(2.0, -0.5), (-1.5, 1.0), (1.0, 0.0)]),
        ("4 hidden layers",
         [(2.0, -0.5), (-1.5, 1.0), (1.0, -0.5), (-0.8, 0.3), (1.0, 0.0)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (title, params) in zip(axes, configs):
        y = [stack_layers(x, params) for x in _X_SCALAR]
        ax.plot(_X_SCALAR, y, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f(x)$")

    fig.suptitle("Piecewise linear functions via stacked ReLU layers")
    fig.tight_layout()

    return _maybe_save_show(fig, save_path, show)


# ==============================================================================
# Block 3 – Vector input, scalar output
# ==============================================================================

def plot_surface(show=False, save_path=None):
    """Plot linear and ReLU surfaces for 2D input → scalar output."""
    grid = torch.linspace(-2, 2, 80)
    X1, X2 = torch.meshgrid(grid, grid, indexing="ij")

    theta_1_2d = torch.tensor([1.0, 0.5])
    theta_0_2d = 0.0

    Z_linear = theta_1_2d[0] * X1 + theta_1_2d[1] * X2 + theta_0_2d
    Z_relu   = relu_tensor(Z_linear)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             subplot_kw={"projection": "3d"})

    for ax, Z, title in zip(
        axes,
        [Z_linear, Z_relu],
        [r"Linear: $\theta_1 \cdot x + \theta_0$", "After ReLU"],
    ):
        ax.plot_surface(X1.numpy(), X2.numpy(), Z.numpy(),
                        cmap="viridis", alpha=0.85)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$f(x)$")
        ax.set_title(title)

    fig.suptitle("2D input → scalar output")
    fig.tight_layout()

    return _maybe_save_show(fig, save_path, show)


# ==============================================================================
# Block 4 – Timing: loop vs. vectorised
# ==============================================================================

def plot_timing(t_loop, t_vec, N, show=False, save_path=None):
    """Bar chart comparing wall-clock time of loop vs. vectorised forward pass.

    Args:
        t_loop: elapsed time for the loop version (seconds)
        t_vec:  elapsed time for the vectorised version (seconds)
        N:      number of samples used in the timing experiment
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Loop", "Vectorised"], [t_loop, t_vec],
                  color=["steelblue", "darkorange"])

    for bar, t in zip(bars, [t_loop, t_vec]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t + max(t_loop, t_vec) * 0.01,
                f"{t:.4f} s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Forward pass: loop vs. vectorised\n(N={N:,} samples)")
    fig.tight_layout()

    return _maybe_save_show(fig, save_path, show)
