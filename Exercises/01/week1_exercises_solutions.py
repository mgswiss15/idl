"""
Introduction to Deep Learning - Week 1 Exercises SOLUTIONS
Neural Network Basics with PyTorch

This file contains complete solutions to all Week 1 exercises.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# ============================================================================
# PART 1: PyTorch Basics
# ============================================================================

print("=" * 70)
print("PART 1: PyTorch Basics")
print("=" * 70)

# Exercise 1.1: Creating and Manipulating Tensors
print("\nExercise 1.1: Creating and Manipulating Tensors")
print("-" * 70)

# (a) Create a tensor from a list
tensor_a = torch.tensor([1, 2, 3, 4, 5])
print(f"Tensor: {tensor_a}")
print(f"Shape: {tensor_a.shape}")
print(f"Data type: {tensor_a.dtype}")

# (b) Create a 3x4 tensor with random values
tensor_b = torch.randn(3, 4)
print(f"\nRandom tensor shape: {tensor_b.shape}")
print(tensor_b)

# (c) Create tensors of ones and zeros
ones_tensor = torch.ones(2, 3)
zeros_tensor = torch.zeros(2, 3)
print(f"\nOnes tensor:\n{ones_tensor}")
print(f"Zeros tensor:\n{zeros_tensor}")

# (d) Reshape tensors
reshaped_1 = tensor_b.reshape(2, 6)
reshaped_2 = tensor_b.reshape(12, 1)
print(f"\nOriginal shape: {tensor_b.shape}")
print(f"Reshaped to (2, 6): {reshaped_1.shape}")
print(f"Reshaped to (12, 1): {reshaped_2.shape}")


# Exercise 1.2: Tensor Operations
print("\n\nExercise 1.2: Tensor Operations")
print("-" * 70)

# (a) Dot product
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.5, -0.3, 0.8])
dot_product = torch.dot(x, w)
print(f"Dot product: {dot_product}")
print(f"Verification: 1*0.5 + 2*(-0.3) + 3*0.8 = {1*0.5 + 2*(-0.3) + 3*0.8}")

# (b) Matrix-vector multiplication
W = torch.randn(3, 2)
x = torch.randn(2)
result = torch.matmul(W, x)  # or W @ x
print(f"\nW shape: {W.shape}")
print(f"x shape: {x.shape}")
print(f"Result shape: {result.shape}")
print(f"Result: {result}")

# (c) Broadcasting
A = torch.randn(2, 3)
B = torch.randn(2, 3)
C = torch.randn(3)
sum_AB = A + B  # Element-wise
sum_AC = A + C  # Broadcasting
print(f"\nA + B shape: {sum_AB.shape}")
print(f"A + C shape: {sum_AC.shape}")
print("C is broadcast to match A's shape!")


# ============================================================================
# PART 2: Activation Functions
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 2: Activation Functions")
print("=" * 70)

# Exercise 2.1: Implement Activation Functions
print("\nExercise 2.1: Implement Activation Functions")
print("-" * 70)


def sigmoid(z):
    """Sigmoid activation: σ(z) = 1 / (1 + exp(-z))"""
    return 1 / (1 + torch.exp(-z))


def tanh(z):
    """Tanh activation: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))"""
    return (torch.exp(z) - torch.exp(-z)) / (torch.exp(z) + torch.exp(-z))


def relu(z):
    """ReLU activation: ReLU(z) = max(0, z)"""
    return torch.maximum(torch.tensor(0.0), z)
    # Alternative: torch.clamp(z, min=0)


# Test implementations
test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {test_input}")
print(f"Sigmoid: {sigmoid(test_input)}")
print(f"Tanh: {tanh(test_input)}")
print(f"ReLU: {relu(test_input)}")

# Compare with PyTorch built-in
print(f"\nCompare with PyTorch:")
print(f"Sigmoid match: {torch.allclose(sigmoid(test_input), torch.sigmoid(test_input))}")
print(f"Tanh match: {torch.allclose(tanh(test_input), torch.tanh(test_input))}")
print(f"ReLU match: {torch.allclose(relu(test_input), torch.relu(test_input))}")


# Exercise 2.2: Visualize Activation Functions
print("\n\nExercise 2.2: Visualize Activation Functions")
print("-" * 70)


def plot_activation_functions():
    """Plot sigmoid, tanh, and ReLU activation functions."""
    z = torch.linspace(-5, 5, 100)
    
    sig_output = sigmoid(z)
    tanh_output = tanh(z)
    relu_output = relu(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z.numpy(), sig_output.numpy(), label='Sigmoid', linewidth=2)
    plt.plot(z.numpy(), tanh_output.numpy(), label='Tanh', linewidth=2)
    plt.plot(z.numpy(), relu_output.numpy(), label='ReLU', linewidth=2)
    plt.xlabel('z', fontsize=12)
    plt.ylabel('Activation', fontsize=12)
    plt.title('Activation Functions', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'activation_functions.png'")
    plt.close()


plot_activation_functions()


# Exercise 2.3: Derivatives of Activation Functions
print("\n\nExercise 2.3: Derivatives of Activation Functions")
print("-" * 70)


def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)


def tanh_derivative(z):
    """Derivative of tanh: tanh'(z) = 1 - tanh²(z)"""
    t = tanh(z)
    return 1 - t**2


def relu_derivative(z):
    """Derivative of ReLU: ReLU'(z) = 1 if z > 0, else 0"""
    return (z > 0).float()


# Plot activation functions and their derivatives
def plot_activations_and_derivatives():
    z = torch.linspace(-5, 5, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sigmoid
    axes[0].plot(z.numpy(), sigmoid(z).numpy(), label='σ(z)', linewidth=2)
    axes[0].plot(z.numpy(), sigmoid_derivative(z).numpy(), label="σ'(z)", 
                linewidth=2, linestyle='--')
    axes[0].set_title('Sigmoid', fontsize=12)
    axes[0].set_xlabel('z')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Tanh
    axes[1].plot(z.numpy(), tanh(z).numpy(), label='tanh(z)', linewidth=2)
    axes[1].plot(z.numpy(), tanh_derivative(z).numpy(), label="tanh'(z)", 
                linewidth=2, linestyle='--')
    axes[1].set_title('Tanh', fontsize=12)
    axes[1].set_xlabel('z')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # ReLU
    axes[2].plot(z.numpy(), relu(z).numpy(), label='ReLU(z)', linewidth=2)
    axes[2].plot(z.numpy(), relu_derivative(z).numpy(), label="ReLU'(z)", 
                linewidth=2, linestyle='--')
    axes[2].set_title('ReLU', fontsize=12)
    axes[2].set_xlabel('z')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activation_derivatives.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'activation_derivatives.png'")
    plt.close()


plot_activations_and_derivatives()


# Exercise 2.4: Vanishing Gradient Problem
print("\n\nExercise 2.4: Vanishing Gradient Problem")
print("-" * 70)

z_values = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
derivatives = sigmoid_derivative(z_values)

print("Sigmoid derivative at different z values:")
for z, deriv in zip(z_values, derivatives):
    print(f"z = {z:6.1f}, σ'(z) = {deriv:.6f}")

print("\nObservation: Vanishing Gradient Problem")
print("-" * 70)
print("When |z| is large, the gradient is very small (~0.000045)")
print("Maximum gradient at z=0: σ'(0) = 0.25")
print("\nWhy is this a problem?")
print("- Gradients multiply across layers (chain rule)")
print("- Small gradients (<0.25) multiply → vanishingly small")
print("- Early layers get almost zero gradient → don't learn!")
print("\nReLU helps: ReLU'(z) = 1 for z > 0 (no saturation)")


# ============================================================================
# PART 3: Forward Propagation by Hand
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 3: Forward Propagation by Hand")
print("=" * 70)

print("\nExercise 3.1 & 3.2: Forward Pass Computation")
print("-" * 70)

# Define network parameters
x = torch.tensor([1.0, 2.0])
W1 = torch.tensor([[0.5, -0.3],
                   [0.8, 0.2]])
b1 = torch.tensor([0.1, -0.2])
w2 = torch.tensor([1.0, -0.5])
b2 = torch.tensor([0.3])

print("Network parameters:")
print(f"Input x: {x}")
print(f"\nHidden layer W1:\n{W1}")
print(f"Hidden layer b1: {b1}")
print(f"\nOutput layer w2: {w2}")
print(f"Output layer b2: {b2}")

# Forward pass
z1 = W1 @ x + b1
a1 = relu(z1)
z2 = w2 @ a1 + b2
output = z2

print(f"\n{'='*70}")
print("FORWARD PASS")
print("="*70)
print(f"Step 1: z1 = W1 @ x + b1 = {z1}")
print(f"        Manual: [0.5*1 + (-0.3)*2 + 0.1, 0.8*1 + 0.2*2 + (-0.2)]")
print(f"        = [{0.5*1 + (-0.3)*2 + 0.1}, {0.8*1 + 0.2*2 + (-0.2)}]")
print(f"\nStep 2: a1 = ReLU(z1) = {a1}")
print(f"\nStep 3: z2 = w2 @ a1 + b2 = {z2}")
print(f"        Manual: 1.0*{a1[0].item()} + (-0.5)*{a1[1].item()} + 0.3")
print(f"        = {1.0*a1[0].item() + (-0.5)*a1[1].item() + 0.3}")
print(f"\nStep 4: output = z2 = {output}")


# ============================================================================
# PART 4: Building a Simple Neural Network
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 4: Building a Simple Neural Network")
print("=" * 70)

# Exercise 4.1: Implement a 2-Layer MLP
print("\nExercise 4.1: Implement a 2-Layer MLP")
print("-" * 70)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Simple 2-layer MLP"""
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass: Linear -> ReLU -> Linear"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# Create model
model = SimpleMLP(input_dim=2, hidden_dim=4, output_dim=1)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")


# Exercise 4.2: Test on XOR Problem
print("\n\nExercise 4.2: Test on XOR Problem")
print("-" * 70)

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("XOR Dataset:")
print("Input  | Target")
print("-" * 20)
for i in range(len(X)):
    print(f"{X[i].numpy()} | {y[i].item():.0f}")

with torch.no_grad():
    predictions = model(X)

print("\nUntrained Network Predictions:")
print("Input  | Prediction | Target | Error")
print("-" * 50)
for i in range(len(X)):
    error = abs(predictions[i].item() - y[i].item())
    print(f"{X[i].numpy()} | {predictions[i].item():10.4f} | {y[i].item():6.0f} | {error:.4f}")

print("\nNote: Predictions are random - network is untrained!")


# Exercise 4.3: Visualize Decision Boundary
print("\n\nExercise 4.3: Visualize Decision Boundary")
print("-" * 70)


def plot_decision_boundary(model, X, y):
    """Plot decision boundary of the model."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], 
        dtype=torch.float32
    )
    
    with torch.no_grad():
        Z = model(grid_points)
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.numpy(), levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Network Output')
    
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='RdYlBu', 
               edgecolors='black', s=200, linewidths=2, zorder=10)
    
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1] + 0.1, f'y={int(y[i].item())}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Decision Boundary (Untrained Network)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'decision_boundary.png'")
    plt.close()


plot_decision_boundary(model, X, y)


# Exercise 4.4: Count Parameters
print("\n\nExercise 4.4: Count Parameters")
print("-" * 70)

input_dim = 2
hidden_dim = 4
output_dim = 1

params_layer1 = input_dim * hidden_dim + hidden_dim
params_layer2 = hidden_dim * output_dim + output_dim
total_params_manual = params_layer1 + params_layer2

print(f"Layer 1 (fc1):")
print(f"  Weights: {input_dim} × {hidden_dim} = {input_dim * hidden_dim}")
print(f"  Biases: {hidden_dim}")
print(f"  Total: {params_layer1}")

print(f"\nLayer 2 (fc2):")
print(f"  Weights: {hidden_dim} × {output_dim} = {hidden_dim * output_dim}")
print(f"  Biases: {output_dim}")
print(f"  Total: {params_layer2}")

print(f"\nTotal parameters (manual): {total_params_manual}")

total_params_pytorch = sum(p.numel() for p in model.parameters())
print(f"Total parameters (PyTorch): {total_params_pytorch}")
print(f"Match: {total_params_manual == total_params_pytorch}")

print("\nDetailed breakdown:")
for name, param in model.named_parameters():
    print(f"{name:15s}: {str(param.shape):15s} -> {param.numel():4d} parameters")


# ============================================================================
# WRAP-UP QUESTIONS - SOLUTIONS
# ============================================================================

print("\n\n" + "=" * 70)
print("WRAP-UP QUESTIONS - SOLUTIONS")
print("=" * 70)

print("""
1. Why can't we use a linear activation function in hidden layers?

ANSWER: If we use linear activations, the entire network becomes equivalent 
to a single linear transformation, regardless of depth:
  f₂(f₁(x)) = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
This is just another linear function! We lose the ability to learn non-linear
decision boundaries and cannot solve problems like XOR.

2. Why is ReLU more popular than sigmoid for hidden layers?

ANSWER: ReLU has several advantages:
  - No vanishing gradient: ReLU'(z) = 1 for z > 0 (constant gradient)
  - Computational efficiency: max(0, z) is faster than exp calculations
  - Sparse activation: ~50% of neurons are zero, leading to sparse representations
  - Empirically better performance: trains faster and achieves better results

Main disadvantage: "dead neurons" when z ≤ 0 always (addressed by Leaky ReLU)

3. How many parameters would a 3-layer MLP have with dimensions [10, 50, 50, 5]?
""")

dims = [10, 50, 50, 5]
total = 0
print("Layer-wise calculation:")
for i in range(len(dims) - 1):
    weights = dims[i] * dims[i+1]
    biases = dims[i+1]
    layer_params = weights + biases
    total += layer_params
    print(f"  Layer {i+1}: {dims[i]:3d} → {dims[i+1]:3d}")
    print(f"    Weights: {dims[i]:3d} × {dims[i+1]:3d} = {weights:5d}")
    print(f"    Biases:  {biases:5d}")
    print(f"    Total:   {layer_params:5d}\n")

print(f"ANSWER: Total parameters = {total}")

print("""
4. Can a single-layer perceptron solve XOR? Why or why not?

ANSWER: No, a single-layer perceptron cannot solve XOR because:
  - Single layer learns: f(x) = w₁x₁ + w₂x₂ + b
  - This defines a LINE (hyperplane) in the input space
  - XOR is NOT linearly separable:
    * Points (0,0) and (1,1) should be on one side (output 0)
    * Points (0,1) and (1,0) should be on the other side (output 1)
    * No single line can separate these two groups!
  
SOLUTION: Need at least one hidden layer with non-linear activation to create
non-linear decision boundaries that can separate XOR classes.
""")


print("\n" + "=" * 70)
print("All exercises completed successfully!")
print("=" * 70)
print("\nGenerated files:")
print("  - activation_functions.png")
print("  - activation_derivatives.png")
print("  - decision_boundary.png")
print("\nKey takeaways:")
print("  ✓ PyTorch tensors and basic operations")
print("  ✓ Activation functions and their derivatives")
print("  ✓ Vanishing gradient problem")
print("  ✓ Forward propagation computation")
print("  ✓ Building neural networks with nn.Module")
print("  ✓ Why depth and non-linearity matter")
