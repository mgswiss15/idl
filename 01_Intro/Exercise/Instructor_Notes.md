# Instructor Notes — Exercise Session 1
## Prerequisites Self-Assessment

**Session:** CW13 — 90 min, immediately after the intro lecture
**Format:** Students work independently, one volunteer presents, group discussion

---

## Timing Overview

| Section | Student work | Discussion | Total |
|---|---|---|---|
| 0 — Intro & framing | — | 3 min | 3 min |
| 1 — Mathematical Notation | 10 min | 7 min | 17 min |
| 2 — Linear Algebra | 10 min | 7 min | 17 min |
| 3 — Calculus & Gradients | 12 min | 8 min | 20 min |
| 4 — Python & PyTorch | 5 min | 5 min | 10 min |
| 5 — Coding & Installation | 12 min | 5 min | 17 min |
| Wrap-up & self-study | — | 5 min | 5 min |
| | | **Total** | **~89 min** |

> Section 3 is the heaviest — if time is tight, trim discussion in Sections 1 and 2.

---

## Block 0 — Introduction (3 min)

Set the tone clearly: this is not a test, there are no grades, and honest self-assessment is the goal. Ask students to annotate their sheet as they go (easy / shaky / unfamiliar). Explain the rhythm: work independently, then one volunteer presents, group discusses.

---

## Section 1 — Mathematical Notation (17 min)

**Where time goes:** Ex. 1 is trivial (1 min). Ex. 2 is fast if students know the notation, slow if they don't — watch for confusion on $f: \mathbb{R}^n \to \mathbb{R}$ and $\forall$. Ex. 3 is the most time-consuming. Ex. 4–7 on composition tend to go quickly except Ex. 6 on $h \circ g$.

**Ask a volunteer to present Ex. 3 and Ex. 6.**

**Key points:**
- Subscripts index elements, superscripts in parentheses $(k)$ index examples — the parentheses distinguish from exponentiation; this distinction will appear everywhere in the course
- $\mathbb{R}^n$ is a space, not a number - $\mathbf{x} \in \mathbb{R}^n$ means $\mathbf{x}$ is a vector of $n$ real numbers
- Composition is right-to-left: $f \circ g$ means apply $g$ first — this surprises many students
- $h \circ g$ in Ex. 6 is not well-defined: $g$ outputs a scalar, $h$ expects a vector — dimension compatibility is not optional
- Ex. 7: connect explicitly to the intro lecture — this is a layer of NN

**Common mistakes:**
- Reading $x^{(k)}$ as exponentiation
- Assuming $f \circ g = g \circ f$
- Not checking domain/codomain compatibility before composing

---

## Section 2 — Linear Algebra (17 min)

**Where time goes:** Ex. 1–2 (dot product and MSE) are quick for most students (~3 min). Ex. 3 (matrix operations) takes longer — $AA^\top$ vs $A^\top A$ is where mistakes cluster. Ex. 4 (shape tracking) is fast for strong students but can stall weaker ones on 4c.

**Ask a volunteer to present Ex. 3b and Ex. 4c.**

**Key points:**
- The dot product $\sum x_i y_i$ is the core computation of every neuron — make this explicit
- $\|\mathbf{x}\|_2^2 = \mathbf{x} \cdot \mathbf{x}$ — the squared norm is a dot product with itself
- $AA^\top \in \mathbb{R}^{2\times2}$ and $A^\top A \in \mathbb{R}^{3\times3}$ — they are both square but different sizes and generally not equal; this is a direct consequence of non-commutativity
- In Ex. 4b, adding $\mathbf{b} \in \mathbb{R}^m$ to $XW \in \mathbb{R}^{n \times m}$ works via broadcasting — PyTorch does this silently, worth flagging now
- Shape tracking is a debugging skill: when a network crashes, shapes are the first thing to check

**Common mistakes:**
- Multiplying matrices in the wrong order — always verify inner dimensions first
- Confusing the shapes of $AA^\top$ and $A^\top A$
- Not writing intermediate shapes step by step in Ex. 4c — insist on this

---

## Section 3 — Calculus & Gradients (20 min)

**Where time goes:** Ex. 1–3 (scalar derivatives) take ~4 min for most students. Ex. 4 (chain rule) is the core of the section — decomposition in 4a takes time, 4b is algebra-heavy, 4c grounds it numerically. Ex. 5–6 (vector/matrix gradients) are the hardest and may not be completed by all students in time — that is fine.

**Ask a volunteer to present Ex. 2 (sigmoid derivative) and Ex. 4a–b.**

**Key points:**
- The sigmoid derivative $\sigma'(x) = \sigma(x)(1-\sigma(x))$ is worth memorising — they will use it repeatedly
- ReLU is undefined at 0: in practice we pick 0 or 1 (subgradient) — mention this briefly, details come later in the course
- The decomposition in Ex. 4a is the key step: $z_i = wx_i + b$, $a_i = \sigma(z_i)$, $r_i = a_i - y_i$, $L = \frac{1}{n}\sum r_i^2$ — students who can do this can do backpropagation
- Make the connection explicit: what they derived in Ex. 4 *is* backpropagation for a one-layer network — PyTorch autograd computes exactly this
- Gradient shape in Ex. 5: $\nabla_\mathbf{w} L \in \mathbb{R}^d$, same shape as $\mathbf{w}$; $\nabla_W L \in \mathbb{R}^{m \times d}$, same shape as $W$ — gradients always match the shape of their parameter
- Ex. 6 is intentionally hard — if only strong students finish it, that is expected

**Common mistakes:**
- Applying the chain rule in the wrong order
- Forgetting that $\partial z_i / \partial w = x_i$ — the derivative w.r.t. the weight is the input
- Treating $\nabla_\mathbf{w} L$ as a scalar

---

## Section 4 — Python & PyTorch Concepts (10 min)

**Where time goes:** Both exercises are short. Ex. 1 (reference semantics) takes ~2 min; Ex. 2 (shape prediction) takes ~3 min. This section should move quickly.

**Ask a volunteer to present both.**

**Key points:**
- Ex. 1: `y = x` does not copy the list — both variables point to the same object; use `y = x.copy()` or `y = list(x)` to avoid this; the same issue arises with tensors and matters when implementing in-place operations
- Ex. 2: `C = A @ B` → shape `(3, 2)`; `D = A.T` → shape `(4, 3)`; `E = A.sum(dim=0)` → shape `(4,)` — summing over dim 0 collapses the rows
- `@` is matrix multiply in Python 3.5+ — not element-wise

**Common mistakes:**
- Predicting `E.shape` as `(3,)` — confusing which dimension is collapsed by `sum(dim=0)`
- Assuming `y = x` creates a copy

---

## Section 5 — Coding & Installation Check (17 min)

**Where time goes:** Ex. 1 (MSE) takes ~5 min for students with working environments. Ex. 2 (gradient) takes ~7 min — the main effort is translating the symbolic result from Section 3 into batched tensor operations. Setup issues will surface here and may take individual attention.

**Do not ask for a volunteer — circulate and check screens instead. Briefly discuss common issues at the end.**

**Key points:**
- Ex. 1: the expected MSE value is known from pen and paper — if the code gives a different number, the bug is either in the tensor creation or the norm computation
- Ex. 2: the cleanest implementation computes all residuals as a batch: `R = (X @ W.T + b) - Y`, then `grad_W = (2/n) * R.T @ X`, `grad_b = (2/n) * R.sum(dim=0)` — students who loop over samples are not wrong but should be nudged toward batched operations
- Shape verification is the self-check: `grad_W.shape == W.shape` and `grad_b.shape == b.shape`
- **Fast finishers:** ask them to verify their gradient using `torch.autograd` and check it matches — do not announce this upfront

**Common issues:**
- dtype mismatch — use `torch.float32` consistently; integer tensors will cause errors
- Forgetting to transpose correctly when computing the batched gradient
- Setup failures (missing PyTorch install) — direct these students to the platform setup guide distributed before CW13

---

## Wrap-up (5 min)

Ask students to look back at their sheet and mark each section green / yellow / red. Distribute or display the self-study reading list:

| Topic | Resource |
|---|---|
| Math notation & linear algebra | *Mathematics for Machine Learning* (Deisenroth et al.) Ch. 1–2 — free at mml-book.github.io |
| Calculus & gradients | *Mathematics for Machine Learning* Ch. 5 |
| All maths | *Deep Learning* (Goodfellow et al.) Ch. 2–4 — free at deeplearningbook.org |
| Python | Official Python tutorial — docs.python.org |
| PyTorch | *Learning the Basics* tutorial — pytorch.org/tutorials |

Close with: *"If you consistently struggled with one section today, invest a few hours there before we reach the topics that depend on it."*
