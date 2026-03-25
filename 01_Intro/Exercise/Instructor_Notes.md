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
| 1 | Mathematical Notation | 8 min | 7 min | 15 min |
| 2 | Linear Algebra | 10 min | 8 min | 18 min |
| 3 | Probability & Statistics | 10 min | 8 min | 18 min |
| 4 | Calculus & Gradients | 10 min | 8 min | 18 min |
| 5 | Python & PyTorch | 10 min | 8 min | 18 min |
| 6 | Wrap-up & self-study pointers | — | 8 min | 8 min |
| | | | **Total** | **~100 min** |

> **Note:** The session now runs ~100 min. Consider trimming 2–3 min from the discussion of one of the later blocks if your slot is strictly 90 min, or drop the PyTorch block if students have already completed the setup task and feel comfortable.

---

## Block 0 — Introduction & Framing (5 min)

**What to say:**
> "This session is not a test. There are no grades. The goal is for *you* to get a realistic picture of where you stand before the course starts in earnest. Be honest with yourself — if you struggle with a block, that is useful information. I will give you reading pointers at the end."

**Logistics:**
- Hand out the worksheet (or display it on screen if digital)
- Explain the rhythm: work independently → one volunteer presents → group discussion
- Encourage students to annotate their own worksheet as they go (mark what felt easy / shaky / unknown)

---

## Block 1 — Mathematical Notation (15 min)

### Exercises for students (8 min)

> The goal here is not computation — it is *reading comprehension*. Students should translate each expression into plain English or a concrete numerical example.

**1.1 — Summation and product notation**

- Compute $\displaystyle\sum_{i=1}^{4} x_i$ and $\displaystyle\prod_{i=1}^{4} x_i$ for $\mathbf{x} = (2, 5, 1, 3)^\top$.
- Let $p_k \geq 0$ for $k = 1, \ldots, K$ and $\displaystyle\sum_{k=1}^{K} p_k = 1$. What kind of mathematical object do the $p_k$ represent? Write an expression for $\displaystyle\sum_{k=1}^{K} k \cdot p_k$ and describe what it computes.

**1.2 — Set and type notation**

Translate each expression into plain English:

- $w \in \mathbb{R}$
- $\mathbf{x} \in \mathbb{R}^{784}$
- $f: \mathbb{R}^n \to \mathbb{R}$
- $\forall i \in \{1, \ldots, n\}$
- $A \in \mathbb{R}^{m \times n}$

**1.3 — Subscripts and superscripts**

Read the following definition carefully:

> Let $x_i^{(k)}$ denote the $i$-th feature of the $k$-th data sample, where $i \in \{1, \ldots, d\}$ and $k \in \{1, \ldots, n\}$. The full dataset is collected into a matrix $X \in \mathbb{R}^{n \times d}$, where $X_{ki} = x_i^{(k)}$.

Now answer:

- What does $x_3^{(1)}$ refer to? Where does it appear in $X$?
- What is the shape of $X$ if you have 200 samples and 8 features? Which dimension is which?
- A classmate writes $x_k^{(i)}$ when they mean "the $i$-th feature of the $k$-th sample." What is wrong, and why does it matter?
- Write an expression for the mean of a feature $i$ across all samples. Use $\sum$ notation. How many such means are there in total?

**1.4 — Function and composition notation**

- If $f(x) = x^2$ and $g(x) = x + 1$, compute $(f \circ g)(3)$ and $(g \circ f)(3)$.
- Let $h: \mathbb{R}^n \to \mathbb{R}$ and $g: \mathbb{R} \to \mathbb{R}$. What are the input and output space of $g \circ h$? What about $h \circ g$?
- A neural network layer is written as $f(\mathbf{x}) = \sigma(W\mathbf{x} + \mathbf{b})$ where $W \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, and $\sigma: \mathbb{R} \to \mathbb{R}$ is applied elementwise. What is the input and output space of $f$? For a second layer $g(\mathbf{y}) = \sigma(V\mathbf{y} + \mathbf{c})$ to be composable as $g \circ f$, what constraints follow on $V$ and $\mathbf{c}$?
- Consider a three-layer network $NN = f_3 \circ f_2 \circ f_1$ with $f_1: \mathbb{R}^{784} \to \mathbb{R}^{128}$, $f_2: \mathbb{R}^{128} \to \mathbb{R}^{64}$, $f_3: \mathbb{R}^{64} \to \mathbb{R}^{10}$. What is the input and ouput space of the NN?

---

### Discussion (7 min)

**Ask a volunteer to present 1.3 and 1.4 (first two points) — these tend to reveal the most confusion.**

**Key points to reinforce:**
- Subscripts index *elements*; superscripts in parentheses $(k)$ index *examples or iterations* — the parentheses distinguish them from exponentiation
- $\mathbb{R}^n$ is not a number — it is a *space*; saying $\mathbf{x} \in \mathbb{R}^n$ means $\mathbf{x}$ is a vector of $n$ real numbers
- $f: A \to B$ is a complete statement — it tells you the domain and codomain, not just the rule
- Composition $f \circ g$ means "apply $g$ first, then $f$" — the order is right-to-left, which surprises students
- Dimension compatibility is not optional — a mismatch crashes the computation, in both math and code

**Common mistakes to flag:**
- Reading $x^{(k)}$ as "$x$ to the power $k$" — the parentheses are the key distinction
- Treating $\sum$ as a single number without thinking about what is varying
- Confusing $W_{ij}$ (a scalar element) with $W$ (the whole matrix)
- Assuming $f \circ g = g \circ f$

**A useful framing to offer:**
> "Mathematical notation is a compression format. Once you can read it fluently, a single line like $\mathbf{y} = \sigma(W\mathbf{x} + \mathbf{b})$ tells you everything about a layer's computation. Struggling to read notation is like struggling to read code — it slows everything else down."

**Self-study pointer if struggling:**
*Mathematics for Machine Learning* (Deisenroth et al.) — Table of symbols and Chapter 1. Also the notation guide in *Deep Learning* (Goodfellow et al.), pages xiii–xix.

---

## Block 2 — Linear Algebra (18 min)

### Exercises for students (10 min)

**2.1 — Vectors and inner products**

- Write the following expression in summation notation: "the sum of all pairwise products $x_i y_i$ for $i = 1, \ldots, n$." Does this expression look familiar? Is there any relation to an $\ell_2$ norm and if so what?
- A simple loss function used in regression is mean squared error $\text{MSE} = \|f(\mathbf{x}) - \mathbf{y}\|_2^2$. Let $f(\mathbf{x}) = 2\mathbf{x}$, $\mathbf{x} = (1, 2, 3)^\top$, and $\mathbf{y} = (1.5, 4.5, 7)^\top$. Compute the MSE. What does a value of zero mean?

**2.2 — Matrix operations**

Let $A = \begin{pmatrix} 1 & 2 & 0 \\ 3 & 1 & 4 \end{pmatrix}$ and $\mathbf{x} = \begin{pmatrix} 1 \\ -1 \\ 2 \end{pmatrix}$.

- Compute $A\mathbf{x}$. What is the shape of the result?
- Compute $A^\top$. What is its shape? Is $AA^\top$ defined? What about $A^\top A$? Compute both.
- Is $A^\top A = AA^\top$? What does this tell you about matrix multiplication in general?

**2.3 — Shape tracking**

Answer without computing — just reason about dimensions:

- Let $W_1 \in \mathbb{R}^{4 \times 3}$, $W_2 \in \mathbb{R}^{2 \times 4}$, $\mathbf{x} \in \mathbb{R}^3$. What is the shape of $W_2 W_1 \mathbf{x}$? What about $W_1^\top W_2^\top$?
- Let $X \in \mathbb{R}^{n \times d}$, $W \in \mathbb{R}^{d \times m}$, $\mathbf{b} \in \mathbb{R}^m$. What is the shape of $XW + \mathbf{b}$? What needs to hold for the addition to be valid?
- Let $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times n}$, $C \in \mathbb{R}^{n \times p}$, $\mathbf{v} \in \mathbb{R}^p$. What is the shape of $A(B + B^\top)C\mathbf{v}$? At each step, write the intermediate shape.

---

### Discussion (8 min)

**Ask a volunteer to present 2.2 and the last bullet of 2.3.**

**Key points to reinforce:**
- Matrix–vector product $A\mathbf{x}$ is the core computation of every neural network layer — every forward pass is a sequence of these
- $A^\top A \neq AA^\top$ in general, and they don't even have the same shape unless $A$ is square — commutativity does not hold for matrix multiplication
- Broadcasting in 2.3: adding $\mathbf{b} \in \mathbb{R}^m$ to $XW \in \mathbb{R}^{n \times m}$ works because $\mathbf{b}$ is broadcast across rows — worth flagging as this is what PyTorch does silently
- Shape tracking is a debugging skill — when a network crashes, the first thing to check is shapes at each layer

**Common mistakes to flag:**
- Multiplying matrices in the wrong order — always check inner dimensions first
- Confusing the shape of $A^\top A$ and $AA^\top$ — they are both square but of different sizes
- Assuming $XW + \mathbf{b}$ is

## Block 3 — Calculus & Gradients (18 min)

### Exercises for students (10 min)

**3.1 — Derivatives**

Compute the derivative with respect to $x$:

- $f(x) = x^3 - 2x^2 + 5$
- $f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$ — show that $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- $f(x) = \max(0, x)$ — what is the derivative? Where is it undefined?

**3.2 — Chain rule**

Consider the following function:

$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n}\left(\sigma(wx_i + b) - y_i\right)^2, \quad \text{where } \sigma(z) = \frac{1}{1+e^{-z}}$$

- Identify the composite structure. Define intermediate variables $z_i$, $a_i$, and $r_i$ such that $L$ can be written as a composition of simple functions.
- Using your decomposition, compute $\frac{\partial L}{\partial w}$ step by step using the chain rule. Write out each intermediate partial derivative explicitly. Use the result from 3.1 that $\sigma'(z) = \sigma(z)(1-\sigma(z))$.
- For $n=2$, $\mathbf{x} = (1, 2)^\top$, $\mathbf{y} = (0, 1)^\top$, $w = 2$, $b = 0$: evaluate $\frac{\partial L}{\partial w}$ numerically. In which direction would you adjust $w$ to decrease $L$?

**3.3 — Gradients of a linear layer**

Consider a single linear layer with weight vector $\mathbf{w} \in \mathbb{R}^d$ and bias $b \in \mathbb{R}$, with MSE loss:

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y_i\right)^2$$

- Define $r_i = \mathbf{w}^\top \mathbf{x}^{(i)} + b - y_i$. Compute $\nabla_\mathbf{w} L$ and $\frac{\partial L}{\partial b}$. What is the shape of $\nabla_\mathbf{w} L$?
- Now extend to a full linear layer with weight matrix $W \in \mathbb{R}^{m \times d}$ and bias vector $\mathbf{b} \in \mathbb{R}^m$, with loss:

$$L(W, \mathbf{b}) = \frac{1}{n}\sum_{i=1}^{n}\|\mathbf{y}^{(i)} - (W\mathbf{x}^{(i)} + \mathbf{b})\|_2^2$$

Define $\mathbf{r}^{(i)} = W\mathbf{x}^{(i)} + \mathbf{b} - \mathbf{y}^{(i)}$. Compute $\nabla_W L$ and $\nabla_\mathbf{b} L$. What are their shapes? How do they relate to the shapes of $W$ and $\mathbf{b}$?

---

### Discussion (8 min)

**Ask a volunteer to present 3.1 (sigmoid derivative) and 3.2 (decomposition and chain rule).**

**Key points to reinforce:**
- The chain rule derivation in 3.2 *is* backpropagation — make this connection explicit: every training step computes exactly these derivatives automatically
- ReLU derivative: undefined at 0 but we pick 0 or 1 in practice (subgradient) — this will come up when implementing activations
- The gradient $\nabla_\mathbf{w} L$ has the same shape as $\mathbf{w}$, and $\nabla_W L$ has the same shape as $W$ — gradients always match the shape of their parameter
- The numerical result in 3.2 tells students *which direction* to move — connect explicitly to gradient descent

**Common mistakes to flag:**
- Applying the chain rule in the wrong order
- Forgetting that $\frac{\partial z_i}{\partial w} = x_i$ — the derivative is the input, not the weight
- Treating $\nabla_\mathbf{w} L$ as a scalar rather than a vector of the same shape as $\mathbf{w}$

**Self-study pointer if struggling:**
*Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 5 (Vector Calculus). Khan Academy calculus for basics.

---

## Block 5 — Python & PyTorch Basics (18 min)

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

## Block 5 — Coding & Installation Check (15 min)

### Exercises for students (12 min)

> No scaffolding is provided — students write the code from scratch. The expected numerical results are already known from Blocks 2 and 3, so they can self-check immediately. If the environment is not set up correctly, this block will surface it.

**5.1 — MSE computation**

Using PyTorch tensors, implement the MSE loss $\text{MSE} = \frac{1}{n}\|\mathbf{f}(\mathbf{x}) - \mathbf{y}\|_2^2$ where $f(\mathbf{x}) = 3\mathbf{x} - 1$ for the following vectors:

$$\mathbf{x} = (1, 2, 3, 4, 5, 6, 7, 8)^\top, \quad \mathbf{y} = (2.1, 5.2, 7.8, 11.1, 14.9, 17.2, 20.8, 23.1)^\top$$

- Create $\mathbf{x}$ and $\mathbf{y}$ as PyTorch tensors
- Compute $\hat{\mathbf{y}} = f(\mathbf{x})$ using tensor operations
- Compute the MSE manually using tensor operations — do not use any PyTorch loss functions
- Print the result and verify it matches your pen and paper expectations

**5.2 — Gradient of a linear layer**

Using the symbolic result from 3.3, implement $\nabla_W L$ and $\nabla_\mathbf{b} L$ manually for the following:

$$W \in \mathbb{R}^{2 \times 3}, \quad \mathbf{b} \in \mathbb{R}^2, \quad n = 4 \text{ samples}$$

$$X = \begin{pmatrix} 1 & 2 & 1 \\ 0 & 1 & 3 \\ 2 & 0 & 1 \\ 1 & 1 & 2 \end{pmatrix}, \quad Y = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}, \quad W = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

- Create all tensors in PyTorch
- Compute the residuals $\mathbf{r}^{(i)} = W\mathbf{x}^{(i)} + \mathbf{b} - \mathbf{y}^{(i)}$ for all samples
- Compute $\nabla_W L$ and $\nabla_\mathbf{b} L$ using your symbolic result from 3.3 — no autograd
- Print the shapes of $\nabla_W L$ and $\nabla_\mathbf{b} L$ and verify they match the shapes of $W$ and $\mathbf{b}$

---

### Discussion (3 min)

**This block is primarily a self-check — keep the discussion short and focused on issues that came up.**

**Key points to reinforce:**
- If the MSE result from 5.1 matches the pen and paper result, the environment is working correctly
- The shape of $\nabla_W L$ must match $W$ and the shape of $\nabla_\mathbf{b} L$ must match $\mathbf{b}$ — if not, something is wrong in the implementation
- Computing gradients manually before using autograd builds understanding of what PyTorch does under the hood — they will appreciate this in the backpropagation lecture

**Common issues to flag:**
- Tensor dtype mismatches — use `torch.float32` consistently
- Shape errors when computing residuals — remind students to check intermediate shapes
- Students who used a PyTorch loss function instead of implementing manually — ask them to redo it

**Note for instructor:** Students who finish early can verify their result for 5.2 using `torch.autograd` and check that the gradients match. Do not announce this upfront — let fast finishers discover it as a bonus.

---

## Block 6 — Wrap-up & Self-Study Pointers (8 min)

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
