# Intro lecture plan

**Opening Demo — "Let's build a classifier" (~20 min)**

**Setup (~2 min)**
- Open Claude on the projector, CIFAR-10 dataset ready to download via torchvision
- Frame it casually: *"Let's start by asking an AI to solve a computer vision problem for us"*

**Prompt 1 — MLP (~4 min)**
- Type live: *"Write me a PyTorch script that trains a neural network to classify CIFAR-10 images into 10 classes. Use a simple MLP architecture. Print the training loss each epoch and plot a few example predictions with their true and predicted labels at the end."*
- Paste the generated script into an editor, run it
- While it trains: *"We asked for an MLP — we'll come back to what that means in week 2"*

**Prompt 2 — CNN (~4 min)**
- Type live: *"Now do the same but use a CNN instead"*
- Paste and run in parallel terminal or after MLP finishes
- While it trains: *"We asked for something called a CNN this time — different architecture, same problem"*

**Results (~5 min)**
- Show training loss curves for both models side by side
- Show example predictions with images — a few correct, a few wrong
- Compare test accuracies: CNN will clearly outperform MLP
- Point at a few misclassified examples: *"Why did it get these wrong? We don't know yet."*

**Punchline (~5 min)**
- *"Both of these work. Claude wrote them in seconds. But let me ask you something — what is a weight? What is a loss? What did 'training' actually do to this network? Why does the CNN beat the MLP? Why did it get these images wrong?"*
- Write on the board: **weights, loss, gradient, backpropagation, convolution, optimization**
- *"These are the words Claude used without explaining. By the end of this course, you will understand every single one of them — well enough to write this yourself from scratch, and well enough to know when it fails and why."*
- *"Today we used AI as a black box. This course is about opening it."*

---

Happy with this? Then we can move to the next section — **What is Deep Learning?**