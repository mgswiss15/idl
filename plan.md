# Plan

| Week | Lecture | Exercise | Homework |
| --- | --- | --- | --- |
| [13. CW](CW13.md): 23.–29.03.2026 | Intro & Course Overview: What is Deep Learning, history, applications, course logistics | Probability Foundations (pen & paper): Random variables & distributions, expectation & variance, joint/marginal/conditional probability, Bayes' theorem, MLE intuition, entropy & cross-entropy | **Pre-course task:** PyTorch environment setup (platform-specific guide provided) | 
| 14. CW: 30.03.–05.04.2026 | Fundamentals I: Perceptron, MLP architecture, forward pass | --- | **HW1 out — Fundamentals:** (1) Derive gradients for a 2-layer MLP by hand, (2) extend forward pass to support arbitrary depth, (3) experiment with activation functions and report effects on training |
| 15. CW: 06.–12.04.2026 | --- | Fundamentals I — Forward Pass from Scratch: Implement linear layers with raw tensors, manual activations (ReLU, sigmoid), stack into a full forward pass, compute loss manually, verify against nn.Linear | |
| 16. CW: 13.–19.04.2026 | Fundamentals II: Loss functions, backpropagation, gradient descent | Fundamentals II — Backprop from Scratch: Derive and implement gradients for the loss, chain rule through activations, gradients of linear layer (∂L/∂W, ∂L/∂b, ∂L/∂x), manual weight update loop, verify against autograd | **HW1 due** |
| 17. CW: 20.–26.04.2026 | Practical Training & Optimization I: Optimizers (SGD, Adam), learning rate, regularization | Optimization: Comparing optimizers, plotting training curves | **HW2 out — Optimization & Regularization:** (1) Implement SGD and Adam from scratch, (2) run learning rate sweep and analyze results, (3) compare dropout vs. L2 regularization on an overfitting experiment |
| 18. CW: 27.04.–03.05.2026 | Practical Training & Optimization II: Batch norm, dropout, weight init, debugging training | Regularization & Tuning: Dropout, batch norm, overfitting experiments | |
| 19. CW: 04.–10.05.2026 | --- | --- | **HW2 due** |
| 20. CW: 11.–17.05.2026 | CNNs & Computer Vision I: Convolution, pooling, CNN architecture | --- | **HW3 out — CNNs:** (1) Implement a convolution operation from scratch using tensor ops, (2) build and train a CNN on a small image dataset, (3) visualize learned filters and analyze what they detect |
| 21. CW: 18.–24.05.2026 | CNNs & Computer Vision II: Deep CNN architectures (LeNet, VGG, ResNet ideas) | CNNs: Building and training a CNN on image data | |
| 22. CW: 25.–31.05.2026 | **First portfolio check:** Sequence Models I: RNNs, vanishing gradients, LSTMs & GRUs | Sequence Models: Implementing an RNN/LSTM for sequence data | **HW3 due** / **HW4 out — Sequence Models:** (1) Derive BPTT gradients for a simple RNN by hand, (2) implement an LSTM cell from scratch, (3) compare RNN vs. LSTM on a sequence task and analyze vanishing gradient behavior |
| 23. CW: 01.–07.06.2026 | Sequence Models II / Attention: Attention mechanism, intro to Transformers | --- | |
| 24. CW: 08.–14.06.2026 | Transformers in Practice: Self-attention, positional encoding, using pretrained models | Transformers: Fine-tuning or applying a small Transformer | **HW4 due** / **HW5 out — Transformers:** (1) Implement scaled dot-product attention from scratch, (2) add positional encoding and verify its effect, (3) fine-tune a small pretrained Transformer and analyze attention maps |
| 25. CW: 15.–21.06.2026 | --- | --- | |
| 26. CW: 22.–28.06.2026 | Deep Generative Models I: Autoencoders, latent space, reconstruction loss | Autoencoders: Building a basic autoencoder in PyTorch | **HW5 due** / **HW6 out — Generative Models:** (1) Derive the ELBO for a VAE by hand, (2) implement a VAE and visualize the latent space, (3) experiment with latent dimensionality and analyze reconstruction quality vs. generation diversity |
| 27. CW: 29.06.–05.07.2026 | Deep Generative Models II: Variational Autoencoders (VAEs), reparameterization trick | VAEs: Implementing a VAE, visualizing the latent space | |
| 28. CW: 06.–12.07.2026 | Course Wrap-up & Outlook: Recap, open problems, what comes next (GANs, diffusion, LLMs) | Open Lab / Project Work: Free coding session, Q&A | **HW6 due** |





