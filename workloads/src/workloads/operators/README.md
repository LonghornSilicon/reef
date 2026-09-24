# Operators

The math behind each kernel in this directory, written the way the code computes it.

**Notation.** $x$ is the input and $y$ the output. $\odot$ multiplies elementwise. $d$ is the width of the last axis. Learned parameters are $W, b, \gamma, \beta, E$.

---

## `linear.py`

### Linear

A matrix multiply plus a bias. With $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ and $b \in \mathbb{R}^{d_\text{out}}$:

```math
y = W x + b
```

## `embedding.py`

### Embedding

A table lookup. With $E \in \mathbb{R}^{V \times d}$ (one row per token id) and token id $i$:

```math
y = E_i
```

## `activation.py`

All apply to each element on its own.

### GELU (tanh approximation)

```math
\mathrm{GELU}(x) = \frac{x}{2} \left( 1 + \tanh\left( \sqrt{2/\pi} \, \left( x + 0.044715 \, x^3 \right) \right) \right)
```

### Softmax

Turns a vector into probabilities that sum to 1. Subtracting the maximum $m = \max_j x_j$ first leaves the result unchanged but keeps $e^{x}$ from overflowing:

```math
\mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
```

## `normalization.py`

### LayerNorm

Subtracts the mean and divides by the standard deviation over the last axis, then applies a gain $\gamma$ and (optionally) a shift $\beta$:

```math
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \qquad
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2, \qquad
y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

## `attention.py`

### GroupedQueryAttention

Inputs are queries $Q$ of shape $(B, h, L_q, d)$ and keys and values $K, V$ of shape $(B, h_{kv}, L_k, d)$. Each of the $h_{kv}$ key/value heads is shared by $h / h_{kv}$ query heads, so $K$ and $V$ are first repeated along the head axis to $h$ heads.

**1. Scores.** How much each query matches each key. The scale is $1/\sqrt{d}$ by default. An optional additive bias $A$ (ALiBi) is added here:

```math
S = \frac{Q K^\top}{\sqrt{d}} + A
```

**2. Mask.** Queries are the last $L_q$ positions, so query $i$ sits at position $q = L_k - L_q + i$. With `causal`, it may attend to key position $j$ only if

```math
j \le q \qquad \text{(causal: no looking ahead)}
```

and, with a sliding window of size $w$, also

```math
j > q - w \qquad \text{(only the } w \text{ most recent positions, including itself)}
```

An optional boolean mask (true where disallowed) is OR-ed in; padding masks arrive this way. Disallowed scores are set to $-\infty$, which the code approximates with the dtype's most negative value, so a fully masked row softmaxes to uniform rather than NaN.

**3. Weights and output.** Each query's scores become probabilities, which mix the values:

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(S) \, V
```
