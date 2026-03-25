# Instructor Notes — CW13 Exercise Session
## Prerequisites Review: Self-Assessment Workshop

**Course:** Introduction to Deep Learning (MSc AI)  
**Session:** CW13 — Exercise (90 min), immediately after the intro lecture  
**Format:** Interactive self-assessment — students work independently, then one student presents, group discussion follows  
**Goal:** Quick refresh of prerequisite knowledge; identify gaps for self-study  

---

## Session Overview

| Block | Topic | Student time | Discussion | Total |
|---|---|---|---|---|
| 0 | Introduction & framing | — | 5 min | 5 min |
| 1 | Linear Algebra | 10 min | 8 min | 18 min |
| 2 | Probability & Statistics | 10 min | 8 min | 18 min |
| 3 | Calculus & Gradients | 10 min | 8 min | 18 min |
| 4 | Python & PyTorch | 10 min | 8 min | 18 min |
| 5 | Wrap-up & self-study pointers | — | 8 min | 8 min |
| | | | **Total** | **~85 min** |

---

## Block 0 — Introduction & Framing (5 min)

**What to say:**
> "This session is not a test. There are no grades. The goal is for *you* to get a realistic picture of where you stand before the course starts in earnest. Be honest with yourself — if you struggle with a block, that is useful information. I will give you reading pointers at the end."

**Logistics:**
- Hand out the worksheet (or display it on screen if digital)
- Explain the rhythm: work independently → one volunteer presents → group discussion
- Encourage students to annotate their own worksheet as they go (mark what felt easy / shaky / unknown)

---

## Block 1 — Linear Algebra (18 min)

### Exercises for students (10 min)

**1.1 — Vector operations**  
Let $\mathbf{a} = (2, -1, 3)^\top$ and $\mathbf{b} = (0, 4, 1)^\top$.

- Compute $\mathbf{a} + \mathbf{b}$, $3\mathbf{a} - \mathbf{b}$
- Compute the dot product $\mathbf{a} \cdot \mathbf{b}$
- Compute $\|\mathbf{a}\|_2$

**1.2 — Matrix multiplication**  
Let
$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

- Compute $A\mathbf{x}$
- What are the dimensions of $A^\top A$? Compute it.

**1.3 — Conceptual**  
- What does it mean for two vectors to be orthogonal?
- If $A$ is an $m \times n$ matrix and $B$ is $n \times p$, what is the shape of $AB$?
- What is the rank of a matrix intuitively?

---

### Discussion (8 min)

**Ask a volunteer to present 1.1 and 1.2 on the board.**

**Key points to reinforce:**
- Matrix–vector product $A\mathbf{x}$: this is the core computation of every neural network layer
- Shape tracking: students who can't track shapes will struggle with implementing networks — emphasise this
- Dot product = weighted sum — fundamental operation in neurons

**Common mistakes to flag:**
- Confusing row vs. column vectors
- Multiplying matrices in the wrong order ($AB \neq BA$ in general)
- Forgetting that matrix multiplication requires inner dimensions to match

**Self-study pointer if struggling:**  
*Mathematics for Machine Learning* (Deisenroth et al., free PDF) — Chapter 2 (Linear Algebra). Also Gilbert Strang's *Introduction to Linear Algebra*.

---

## Block 2 — Probability & Statistics (18 min)

### Exercises for students (10 min)

**2.1 — Probability basics**  
A dataset has 70 positive and 30 negative examples.

- What is $P(\text{positive})$?
- If you draw two examples independently, what is $P(\text{both positive})$?
- Apply Bayes' theorem: if a classifier outputs "positive" 90% of the time when the true label is positive, and 20% of the time when the true label is negative, what is $P(\text{true positive} \mid \text{predicted positive})$?

**2.2 — Distributions**  
- Sketch a Gaussian distribution $\mathcal{N}(0, 1)$ and $\mathcal{N}(0, 4)$ on the same axes. What changes?
- A coin flip has $P(\text{heads}) = 0.6$. What distribution models this? What are its mean and variance?

**2.3 — Cross-entropy**  
The true label is $y = 1$. A model outputs $\hat{p} = 0.9$.

- Compute the binary cross-entropy loss: $-[y \log \hat{p} + (1-y) \log(1-\hat{p})]$
- What happens to the loss if the model outputs $\hat{p} = 0.1$ instead?

---

### Discussion (8 min)

**Ask a volunteer to present 2.1 and 2.3.**

**Key points to reinforce:**
- Bayes' theorem: the prior matters — even a good classifier can give misleading posteriors on imbalanced data
- Cross-entropy: connect explicitly to what they will use as the loss function in the course
- $\log(0)$ is undefined — this is why we clip predictions in practice

**Common mistakes to flag:**
- Treating probability and likelihood as synonyms
- Not applying the log carefully (forgetting the minus sign, computing $\log(0)$)
- Confusing $P(A|B)$ and $P(B|A)$ — the classic base rate neglect

**Self-study pointer if struggling:**  
*Probability Theory: The Logic of Science* (Jaynes) for depth, or *Deep Learning* book (Goodfellow et al.) Chapter 3 for the course-relevant subset.

---

## Block 3 — Calculus & Gradients (18 min)

### Exercises for students (10 min)

**3.1 — Derivatives**  
Compute the derivative with respect to $x$:

- $f(x) = x^3 - 2x^2 + 5$
- $f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$ — show that $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- $f(x) = \max(0, x)$ — what is the derivative? Where is it undefined?

**3.2 — Chain rule**  
Let $z = wx + b$, $a = \sigma(z)$, $L = (a - y)^2$.

- Compute $\frac{\partial L}{\partial w}$ using the chain rule step by step.
- Write out each intermediate partial derivative.

**3.3 — Gradients**  
Let $f(\mathbf{w}) = \mathbf{w}^\top \mathbf{w}$ for $\mathbf{w} \in \mathbb{R}^n$.

- Compute $\nabla_\mathbf{w} f$.
- In which direction does $f$ increase fastest?

---

### Discussion (8 min)

**Ask a volunteer to present 3.1 (sigmoid derivative) and 3.2.**

**Key points to reinforce:**
- The chain rule derivation in 3.2 *is* backpropagation — make this connection explicit
- ReLU derivative: undefined at 0 but we just pick 0 or 1 in practice (subgradient)
- The gradient points in the direction of steepest ascent; we go *opposite* to it (gradient descent)

**Common mistakes to flag:**
- Applying chain rule in the wrong order
- Forgetting that $\partial z / \partial w = x$ (the input, not the weight)
- Treating the gradient of a scalar w.r.t. a vector as a scalar

**Self-study pointer if struggling:**  
*Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 5 (Vector Calculus). Khan Academy calculus for basics.

---

## Block 4 — Python & PyTorch Basics (18 min)

### Exercises for students (10 min)

> Students should attempt these mentally or on paper — no laptop needed. The goal is conceptual familiarity, not syntax recall.

**4.1 — Python**  
What does the following return, and why?

```python
x = [1, 2, 3]
y = x
y.append(4)
print(x)
```

What would you change to avoid this behaviour?

**4.2 — PyTorch tensors**  
Without running the code, predict the output shape:

```python
import torch
A = torch.ones(3, 4)
B = torch.ones(4, 2)
C = A @ B          # shape?
D = A.T            # shape?
E = A.sum(dim=0)   # shape?
```

**4.3 — Gradient computation**  
```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
```

- What is `x.grad`?
- What does `requires_grad=True` tell PyTorch to do?
- Why is this relevant to training neural networks?

---

### Discussion (8 min)

**Ask a volunteer to present 4.1 and 4.3.**

**Key points to reinforce:**
- 4.1: mutable default arguments and reference semantics — a source of subtle bugs
- Shape tracking in 4.2: make students verify mentally by matching inner dimensions
- 4.3: `x.grad = 6.0` because $\frac{d}{dx}x^2 = 2x = 6$. This is exactly what PyTorch's autograd computes during backpropagation — connect to Block 3

**Common mistakes to flag:**
- Confusing `tensor.shape` with `len(tensor)`
- Calling `.backward()` more than once without zeroing gradients (will matter in training)
- Not knowing that `@` is matrix multiply in Python 3.5+

**Self-study pointer if struggling:**  
Official PyTorch tutorials: *Learning the Basics* and *Autograd tutorial* at pytorch.org. The setup guide distributed before CW13 also covers basic tensor operations.

---

## Block 5 — Wrap-up & Self-Study Pointers (8 min)

**What to say:**
> "Take a moment to look back at your worksheet. Mark each block: green (comfortable), yellow (shaky), red (struggled). This is your personal study map for the first two weeks."

**Distribute or display the self-study reading list:**

| Topic | Resource | Where |
|---|---|---|
| Linear algebra | *Mathematics for Machine Learning* Ch. 2 | mml-book.github.io (free) |
| Probability & stats | *Mathematics for Machine Learning* Ch. 6 | mml-book.github.io (free) |
| Calculus & gradients | *Mathematics for Machine Learning* Ch. 5 | mml-book.github.io (free) |
| Python fundamentals | Official Python tutorial | docs.python.org |
| PyTorch basics | PyTorch *Learning the Basics* tutorial | pytorch.org/tutorials |
| Deep learning maths (all) | *Deep Learning* (Goodfellow et al.) Ch. 2–4 | deeplearningbook.org (free) |

**Closing message:**
> "You don't need to be an expert in all of this before next week. But if you consistently struggled with one block today, invest a few hours there before we reach the topics that depend on it. I am available in office hours if you have questions."

---

## Instructor Checklist

- [ ] Print or distribute the student worksheet before the session
- [ ] Prepare the self-study reading list as a handout or post it on the e-learning platform
- [ ] Have solutions ready (for your reference only — not distributed)
- [ ] Write block headings on the board to signal transitions
- [ ] Note which blocks generated the most difficulty — useful signal for pacing the first few lectures

---

## Notes on Interactivity

The session works best if you actively **cold-call** rather than waiting for volunteers after the first block. A light touch ("Who got a different answer for 3.2? Let's compare.") keeps the discussion lively without putting students on the spot. Treat wrong answers as teaching moments, not corrections.

If a block goes quickly, use the spare time to ask a deeper question: *"Where do you think we will use dot products in a neural network?"* rather than rushing to the next block.
