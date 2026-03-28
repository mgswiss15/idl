# Introduction to Deep Learning — Course Plan
## THWS Masters in AI, First Semester

---

## Schedule overview

13 lectures + 11 exercise sessions, CW13–CW28.

---

## Lecture arc

| # | Lecture | Block |
|---|---------|-------|
| L1 | Introduction & course overview | — |
| L2 | Perceptron, MLPs & activations | Fundamentals & MLPs |
| L3 | Loss functions & MLE | Fundamentals & MLPs |
| L4 | Backpropagation | Fundamentals & MLPs |
| L5 | PyTorch: tensors, autograd, training loop & evaluation | Practical Training & Optimization |
| L6 | Optimizers, schedules & initialization | Practical Training & Optimization |
| L7 | Regularization & generalization | Practical Training & Optimization |
| L8 | CNN architecture & convolution | CNNs & Computer Vision |
| L9 | Modern CNNs & vision tasks | CNNs & Computer Vision |
| L10 | RNNs, LSTMs & attention | Sequence Models |
| L11 | Transformers & language models | Sequence Models |
| L12 | VAEs & GANs | Deep Generative Modeling |
| L13 | Diffusion models, survey & outlook | — |

---

## Block breakdown

## Block 1 — Fundamentals & MLPs (L2–L4)

**L2 — Perceptron, MLPs & activations**
- Biological neuron → artificial neuron
- Perceptron: weights, bias, step function, geometric interpretation
- Perceptron learning rule & convergence; XOR as the breaking point
- From perceptron to MLP: stacking layers, hidden representations
- Activation functions: sigmoid, tanh, ReLU — motivation, properties, dying ReLU
- Universal approximation theorem (statement, no proof)
- Notation summary slide (W, b, z, a, σ) — end of lecture

**L3 — Loss functions & MLE**
- The learning problem framed probabilistically
- MLE as a principled foundation
- Cross-entropy loss (binary and multiclass), MSE
- Connecting loss to optimization: what does minimizing loss mean geometrically?

**L4 — Backpropagation**
- Computation graphs
- Chain rule: scalar → vector → general
- Backprop: local gradients, message-passing view
- Worked example: 2-layer MLP end-to-end
- What students will implement in the exercise

---

## Block 2 — Practical Training & Optimization (L5–L7)

**L5 — PyTorch: tensors, autograd, training loop & evaluation**
- Tensors, `requires_grad`, dynamic computation graph
- Autograd: `.backward()`, `.grad`, `.zero_grad()`, `.step()`, `torch.no_grad()`
- `nn.Module` as a parameter container
- Training loop: epochs, batches, `Dataset`, `DataLoader`, forward → loss → backward → step
- Model evaluation: `model.eval()`, `model.train()`, metrics

**L6 — Optimizers, schedules & initialization**
- Gradient descent: batch, mini-batch, stochastic — tradeoffs
- Momentum, RMSProp, Adam — derivation and intuition
- Learning rate schedules: warmup, cosine annealing, step decay — `torch.optim.lr_scheduler`
- Gradient clipping: `nn.utils.clip_grad_norm_`
- Weight initialization: Xavier, He — `nn.init`

**L7 — Regularization & generalization**
- Bias–variance tradeoff
- L1/L2 regularization — `weight_decay` in `torch.optim`
- Dropout — `nn.Dropout`, train vs eval mode
- Batch normalization — `nn.BatchNorm1d/2d`, behavior during train vs eval
- Early stopping — practical implementation pattern
- Data augmentation (brief) — `torchvision.transforms`

---

## Block 3 — CNNs & Computer Vision (L8–L9)

**L8 — CNN architecture & convolution**
- Motivation: why convolution for images?
- Convolution operation: kernels, stride, padding
- Pooling: max, average
- Receptive field
- PyTorch: `nn.Conv2d`, `nn.MaxPool2d`, `torchvision.transforms`, `torchvision.datasets`
- Putting it together: LeNet-style architecture, small CNN implementation end-to-end

**L9 — Modern CNNs & vision tasks**
- VGG, ResNet, skip connections
- Transfer learning & fine-tuning: when and why, freezing layers, fine-tuning strategies — `torchvision.models`
- Object detection: problem formulation, YOLO, Faster R-CNN (conceptual)
- Semantic segmentation: problem formulation, U-Net (conceptual)
- Other convolution types: depthwise, grouped — brief mention

---

## Block 4 — Sequence Models (L10–L11)

**L10 — RNNs & LSTMs**
- Recurrent networks: unrolling, BPTT
- Vanishing/exploding gradients
- LSTMs: gating mechanism
- GRU: brief mention
- PyTorch: `nn.RNN`, `nn.LSTM`, packing/padding sequences

**L11 — Attention & transformers**
- Attention as a solution to the bottleneck problem
- Self-attention, scaled dot-product attention
- Multi-head attention
- Positional encoding
- Transformer architecture
- Pre-training & fine-tuning: BERT, GPT paradigms (conceptual)
- Scaling laws — brief mention
- HuggingFace — brief pointer, `pipeline` API
- PyTorch: `nn.TransformerEncoder`

---

## Block 5 — Deep Generative Modeling (L12–L13)

**L12 — Autoencoders, VAEs & GANs**
- Autoencoder: architecture, reconstruction loss, latent space visualization
- From AE to VAE: why the latent space needs structure
- VAE: ELBO, reparameterization trick, architecture & training
- PyTorch: VAE implementation, `torch.distributions`
- GAN: adversarial training framework, generator & discriminator
- GAN training dynamics, mode collapse
- Variants: conditional GAN, DCGAN (brief)

**L13 — Diffusion models & research frontier**
- Score matching intuition
- DDPM: forward & reverse process, connection to VAEs
- Where generative modeling stands today
- Open problems and research frontiers in deep learning broadly
- What's next: emerging architectures, scaling, multimodal models

---

## Homework assignments

One per topic block, combining:
- From-scratch PyTorch implementation
- Theoretical/math problems
- Small experiment with written analysis

| HW | Block | Core task |
|----|-------|-----------|
| HW1 | Fundamentals | Implement backprop manually (no autograd) |
| HW2 | Practical Training | Train & tune an MLP; optimizer & regularization ablations |
| HW3 | CNNs | Build and train a CNN on image data |
| HW4 | Sequences | Implement an RNN/LSTM; experiment with a transformer |
| HW5 | Generative | Train a VAE or GAN; analyze latent space |

---

## Pedagogical principles

- Notation fixed in L2 and used consistently throughout
- Students implement backpropagation manually before using `nn.Module`
- Probability concepts introduced just-in-time (MLE in L3, not upfront)
- Exercises interleaved with theory, not front-loaded
- Python scripts over Jupyter notebooks
- PyTorch tensors directly (not NumPy)
