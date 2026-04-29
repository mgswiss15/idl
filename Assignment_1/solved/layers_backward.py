"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - layers with backward pass

Magda Gregorová, April 2026

SOLUTION VERSION — do not distribute to students.
"""

import torch
from functions import linear_ggrad, relu_ggrad


class Linear:
    """Linear layer with forward and backward pass."""

    def __init__(self, theta_1, theta_0):
        self.theta_1 = theta_1
        self.theta_0 = theta_0

    def forward(self, ins):
        self.ins  = ins
        self.outs = ins @ self.theta_1.T + self.theta_0
        return self.outs

    def backward(self, gout):
        self.theta_1.g, self.theta_0.g, self.ins.g = linear_ggrad(
            gout, self.ins, self.theta_1, self.theta_0
        )
        return self.ins.g


class ReLU:
    """ReLU non-linearity with forward and backward pass."""

    def forward(self, ins):
        self.ins  = ins
        self.outs = ins.clamp(0)
        return self.outs

    def backward(self, gout):
        self.ins.g = relu_ggrad(gout, self.ins)
        return self.ins.g


class Model:
    """Neural network model with forward and backward pass."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, ins):
        out = ins
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, gout):
        for layer in reversed(self.layers):
            gout = layer.backward(gout)
        return gout
