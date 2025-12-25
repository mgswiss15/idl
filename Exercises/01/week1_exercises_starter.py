"""
Introduction to Deep Learning - Week 1 Exercises
Neural Network Basics with PyTorch

Instructions:
- Complete the TODOs in each section
- Run the script to test your implementations
- Check your results against expected outputs
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
# TODO: Create a tensor from [1, 2, 3, 4, 5]
tensor_a = None  # Replace with your code

if tensor_a is not None:
    print(f"Tensor: {tensor_a}")
    print(f"Shape: {tensor_a.shape}")
    print(f"Data type: {tensor_a.dtype}")

# (b) Create a 3x4 tensor with random values from standard normal distribution
# TODO: Use torch.randn()
tensor_b = None  # Replace with your code

if tensor_b is not None:
    print(f"\nRandom tensor shape: {tensor_b.shape}")
    print(tensor_b)

# (c) Create tensors of ones and zeros
# TODO: Create 2x3 tensors
ones_tensor = None  # Replace with your code
zeros_tensor = None  # Replace with your code

# (d) Reshape tensors
# TODO: Reshape tensor_b to (2, 6) then to (12, 1)
if tensor_b is not None:
    reshaped_1 = None  # tensor_b reshaped to (2, 6)
    reshaped_2 = None  # tensor_b reshaped to (12, 1)


# Exercise 1.2: Tensor Operations
print("\n\nExercise 1.2: Tensor Operations")
print("-" * 70)

# (a) Dot product
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.5, -0.3, 0.8])

# TODO: Compute dot product
dot_product = None  # Replace with your code

if dot_product is not None:
    print(f"Dot product: {dot_product}")

# (b) Matrix-vector multiplication
W = torch.randn(3, 2)
x = torch.randn(2)

# TODO: Compute W @ x using torch.matmul() or @ operator
result = None  # Replace with your code

if result is not None:
    print(f"\nW shape: {W.shape}")
    print(f"x shape: {x.shape}")
    print(f"Result shape: {result.shape}")

# (c) Broadcasting
A = torch.randn(2, 3)
B = torch.randn(2, 3)
C = torch.randn(3)

# TODO: Element-wise addition
sum_AB = None  # A + B
sum_AC = None  # A + C (broadcasting)


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
    """
    Sigmoid activation function: σ(z) = 1 / (1 + exp(-z))
    
    Args:
        z: Input tensor
    Returns:
        Output tensor with sigmoid applied element-wise
    """
    # TODO: Implement sigmoid
    pass


def tanh(z):
    """
    Tanh activation function: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    
    Args:
        z: Input tensor
    Returns:
        Output tensor with tanh applied element-wise
    """
    # TODO: Implement tanh
    pass


def relu(z):
    """
    ReLU activation function: ReLU(z) = max(0, z)
    
    Args:
        z: Input tensor
    Returns:
        Output tensor with ReLU applied element-wise
    """
    # TODO: Implement ReLU
    # Hint: Use torch.maximum() or torch.clamp()
    pass


# Test your implementations
test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {test_input}")

if sigmoid(test_input) is not None:
    print(f"Sigmoid: {sigmoid(test_input)}")
if tanh(test_input) is not None:
    print(f"Tanh: {tanh(test_input)}")
if relu(test_input) is not None:
    print(f"ReLU: {relu(test_input)}")


# Exercise 2.2: Visualize Activation Functions
print("\n\nExercise 2.2: Visualize Activation Functions")
print("-" * 70)


def plot_activation_functions():
    """Plot sigmoid, tanh, and ReLU activation functions."""
    z = torch.linspace(-5, 5, 100)
    
    # TODO: Apply activation functions
    sig_output = None  # sigmoid(z)
    tanh_output = None  # tanh(z)
    relu_output = None  # relu(z)
    
    if sig_output is not None and tanh_output is not None and relu_output is not None:
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


# Uncomment to generate plot
# plot_activation_functions()


# Exercise 2.3: Derivatives of Activation Functions
print("\n\nExercise 2.3: Derivatives of Activation Functions")
print("-" * 70)


def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
    # TODO: Implement
    pass


def tanh_derivative(z):
    """Derivative of tanh: tanh'(z) = 1 - tanh²(z)"""
    # TODO: Implement
    pass


def relu_derivative(z):
    """Derivative of ReLU: ReLU'(z) = 1 if z > 0, else 0"""
    # TODO: Implement
    # Hint: Use (z > 0).float() to convert boolean to float
    pass


# Exercise 2.4: Vanishing Gradient Problem
print("\n\nExercise 2.4: Vanishing Gradient Problem")
print("-" * 70)

z_values = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])

# TODO: Compute sigmoid derivatives
# derivatives = sigmoid_derivative(z_values)

# Uncomment when implemented:
# print("Sigmoid derivative at different z values:")
# for z, deriv in zip(z_values, derivatives):
#     print(f"z = {z:6.1f}, σ'(z) = {deriv:.6f}")


# ============================================================================
# PART 3: Forward Propagation by Hand
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 3: Forward Propagation by Hand")
print("=" * 70)

# Exercise 3.1 & 3.2: Manual Computation and Verification
print("\nExercise 3.1 & 3.2: Forward Pass Computation")
print("-" * 70)

# Define network parameters
x = torch.tensor([1.0, 2.0])

# Hidden layer weights and bias
W1 = torch.tensor([[0.5, -0.3],
                   [0.8, 0.2]])
b1 = torch.tensor([0.1, -0.2])

# Output layer weights and bias
w2 = torch.tensor([1.0, -0.5])
b2 = torch.tensor([0.3])

print("Network parameters:")
print(f"Input x: {x}")
print(f"\nHidden layer W1:\n{W1}")
print(f"Hidden layer b1: {b1}")
print(f"\nOutput layer w2: {w2}")
print(f"Output layer b2: {b2}")

# TODO: Compute forward pass step by step
# Step 1: Pre-activation of hidden layer
z1 = None  # W1 @ x + b1

# Step 2: Activation of hidden layer (ReLU)
a1 = None  # relu(z1)

# Step 3: Pre-activation of output layer
z2 = None  # w2 @ a1 + b2

# Step 4: Output (no activation)
output = None  # z2

# Uncomment when implemented:
# print(f"\nz1 (pre-activation hidden): {z1}")
# print(f"a1 (activation hidden): {a1}")
# print(f"z2 (pre-activation output): {z2}")
# print(f"Final output: {output}")


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
        """
        Simple 2-layer MLP
        
        Args:
            input_dim: Dimension of input
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output
        """
        super(SimpleMLP, self).__init__()
        
        # TODO: Initialize layers
        # Hint: Use nn.Linear(in_features, out_features)
        self.fc1 = None  # First linear layer
        self.fc2 = None  # Second linear layer
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # TODO: Implement forward pass
        # Layer 1: Linear -> ReLU
        # Layer 2: Linear (no activation)
        
        pass


# Create model
# TODO: Uncomment when SimpleMLP is implemented
# model = SimpleMLP(input_dim=2, hidden_dim=4, output_dim=1)
# print(model)
# print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")


# Exercise 4.2: Test on XOR Problem
print("\n\nExercise 4.2: Test on XOR Problem")
print("-" * 70)

# Create XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("XOR Dataset:")
print("Input  | Target")
print("-" * 20)
for i in range(len(X)):
    print(f"{X[i].numpy()} | {y[i].item():.0f}")

# TODO: Uncomment when model is implemented
# with torch.no_grad():
#     predictions = model(X)
#
# print("\nUntrained Network Predictions:")
# for i in range(len(X)):
#     print(f"Input: {X[i].numpy()}, Prediction: {predictions[i].item():.4f}, Target: {y[i].item():.0f}")


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
    
    # TODO: Create input tensor from mesh grid
    # Hint: Stack xx.ravel() and yy.ravel(), then convert to torch tensor
    grid_points = None  # Shape should be (n_points, 2)
    
    if grid_points is not None:
        # Get predictions
        with torch.no_grad():
            Z = model(grid_points)
        
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z.numpy(), levels=20, cmap='RdYlBu', alpha=0.8)
        plt.colorbar(label='Network Output')
        
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='RdYlBu', 
                   edgecolors='black', s=200, linewidths=2)
        
        plt.xlabel('x₁', fontsize=12)
        plt.ylabel('x₂', fontsize=12)
        plt.title('Decision Boundary (Untrained Network)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
        print("Plot saved to 'decision_boundary.png'")
        plt.close()


# TODO: Uncomment when model is implemented
# plot_decision_boundary(model, X, y)


# Exercise 4.4: Count Parameters
print("\n\nExercise 4.4: Count Parameters")
print("-" * 70)

input_dim = 2
hidden_dim = 4
output_dim = 1

# TODO: Calculate number of parameters manually
# Layer 1: input_dim * hidden_dim + hidden_dim (weights + biases)
# Layer 2: hidden_dim * output_dim + output_dim (weights + biases)

params_layer1 = None  # Replace with calculation
params_layer2 = None  # Replace with calculation
total_params_manual = None  # Replace with calculation

# Uncomment when implemented:
# print(f"Layer 1 parameters: {params_layer1}")
# print(f"Layer 2 parameters: {params_layer2}")
# print(f"Total parameters (manual): {total_params_manual}")
#
# # Verify with PyTorch
# total_params_pytorch = sum(p.numel() for p in model.parameters())
# print(f"Total parameters (PyTorch): {total_params_pytorch}")
# print(f"Match: {total_params_manual == total_params_pytorch}")


# ============================================================================
# WRAP-UP QUESTIONS
# ============================================================================

print("\n\n" + "=" * 70)
print("WRAP-UP QUESTIONS")
print("=" * 70)

print("""
Answer these questions based on what you've learned:

1. Why can't we use a linear activation function in hidden layers?
   Your answer: 

2. Why is ReLU more popular than sigmoid for hidden layers?
   Your answer: 

3. How many parameters would a 3-layer MLP have with dimensions [10, 50, 50, 5]?
   Your answer: 

4. Can a single-layer perceptron solve XOR? Why or why not?
   Your answer: 
""")

# Question 3: Calculate parameters for a 3-layer MLP
dims = [10, 50, 50, 5]

# TODO: Calculate total parameters
# total = 0
# for i in range(len(dims) - 1):
#     layer_params = None  # Calculate params for layer i
#     total += layer_params
#     print(f"Layer {i+1}: {dims[i]} -> {dims[i+1]}, Parameters: {layer_params}")
#
# print(f"\nTotal parameters: {total}")


print("\n\n" + "=" * 70)
print("Exercises completed!")
print("=" * 70)
print("\nNext steps:")
print("1. Complete all TODO sections")
print("2. Uncomment the test code to verify your implementations")
print("3. Run the script and check outputs")
print("4. Experiment with different values and parameters")
print("\nGood luck!")
