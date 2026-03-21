# Intro plan

| Week | Lecture | Exercise | Homework |
| --- | --- | --- | --- |
| Intro | Intro & Course Overview: What is Deep Learning, history, applications, course logistics | Probability Foundations (pen & paper): Random variables & distributions, expectation & variance, joint/marginal/conditional probability, Bayes' theorem, MLE intuition, entropy & cross-entropy | **Pre-course task:** PyTorch environment setup (platform-specific guide provided) | 


## Lecture

**Core strategy:** Lead with a live demo rather than theory. Open Claude, prompt it to generate a CIFAR-10 classifier, run it, and let it work. The punchline — *"this works, but we have no idea what happened"* — reframes the entire course as an intellectual journey: understanding every piece of that black box. History and logistics follow, but the demo does the motivational heavy lifting.

**Content:**
- **Opening demo** (~20 min) — Live Claude prompt → CIFAR-10 classifier → it trains → punchline
- **What is Deep Learning?** (~15 min) — AI/ML landscape, why now
- **Historical milestones** (~20 min) — Perceptron to AlexNet to transformers
- **Application areas** (~10 min) — brief, demo already motivated this
- **Course roadmap & logistics** (~15 min) — schedule, tools, assessment, setup task
- **Buffer / Q&A** (~10 min)

**Key pedagogical moves:**
- Start with the end result, then spend the semester reverse-engineering it
- Use the demo as a recurring reference point throughout the course (*"remember week 1?"*)
- AlexNet milestone echoes the demo — same idea, but it changed the world
- Close with *"by the end of this course you will understand every line Claude wrote today"*

## Exercise

**Format:** Pen & paper, 90 min, guided discovery. Each block opens with a puzzle or paradox that creates genuine surprise before any concept is named. Discovery is triggered through a rotating mix of: students predicting first, brief pair discussion, and Socratic questioning. Solutions are revealed after discussing common mistakes.

**The four anchoring paradoxes:**
1. **Biased coin** — 8 heads in 10 flips: is the coin fair? Motivates the gap between observed data and the true distribution.
2. **Die game** — how much would you pay to play once vs. 1000 times? Motivates expectation and variance from pure intuition.
3. **n vs. n-1** — two people compute variance of the same 5 numbers and both are right. Motivates the population vs. sample distinction and why it matters for generalization.
4. **Monty Hall** — the classic. Resolved using a probability table, motivating conditional probability and Bayes' theorem as natural tools for reasoning under uncertainty.

**Key pedagogical moves:**
- Students commit to an answer in writing before any explanation — prevents passive listening
- Confusion is deliberately held for a moment before resolution
- Each block ends with a forward bridge to deep learning concepts later in the course
- MLE and entropy deliberately left out — handled just-in-time when loss functions appear in week 16



