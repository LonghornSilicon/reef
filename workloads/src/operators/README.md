# Operators

The math behind each kernel in this directory, written the way the code computes it.

**Notation.** $x$ is the input and $y$ the output. $\odot$ multiplies elementwise. $d$ is the width of the last axis. Learned parameters are $W, b, w, \gamma, \beta, E$. Tensor shapes are written like $(B, C, H, W)$: batch, channels, height, width.

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

## `positional.py`

### RotaryEmbedding (RoPE)

Encodes a token's position $n$ by rotating pairs of query and key features through position-dependent angles. For rotary width $r$ (the head width by default), base $\theta$ and scaling factor $s$, there is one frequency per pair $k = 0, \dots, r/2 - 1$:

```math
f_k = \frac{1}{s \cdot \theta^{2k/r}}
```

The frequencies can also be supplied directly, which is how Llama 3's smoothed table enters. Split the rotated slice into halves $x^{(1)}$ and $x^{(2)}$. Each pair $\big(x^{(1)}_k, x^{(2)}_k\big)$ is rotated by angle $n f_k$:

```math
\begin{bmatrix} y^{(1)}_k \\ y^{(2)}_k \end{bmatrix}
=
\begin{bmatrix} \cos(n f_k) & -\sin(n f_k) \\ \sin(n f_k) & \cos(n f_k) \end{bmatrix}
\begin{bmatrix} x^{(1)}_k \\ x^{(2)}_k \end{bmatrix}
```

The code writes the same rotation as `x * cos + rotate_half(x) * sin`, where `rotate_half` maps $[x^{(1)}, x^{(2)}]$ to $[-x^{(2)}, x^{(1)}]$. Features past $r$ pass through unchanged.

## `activation.py`

All apply to each element on its own.

### ReLU

```math
\mathrm{ReLU}(x) = \max(0, x)
```

### SiLU

```math
\mathrm{SiLU}(x) = x \cdot \sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
```

### Sigmoid

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

### GELU (tanh approximation)

```math
\mathrm{GELU}(x) = \frac{x}{2} \left( 1 + \tanh\left( \sqrt{2/\pi} \, \left( x + 0.044715 \, x^3 \right) \right) \right)
```

### Softmax

Turns a vector into probabilities that sum to 1. Subtracting the maximum $m = \max_j x_j$ first leaves the result unchanged but keeps $e^{x}$ from overflowing:

```math
\mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
```

## `dropout.py`

### Dropout

In training, each element is zeroed with probability $p$, and the survivors are scaled up so the expected value is unchanged. $M_i$ is 1 with probability $1 - p$, else 0:

```math
y = \frac{x \odot M}{1 - p}
```

In eval mode, $y = x$.

## `normalization.py`

### RMSNorm

Divides by the root mean square over the last axis, then applies a learned gain $w$:

```math
y = w \odot \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}}
```

### LayerNorm

Subtracts the mean and divides by the standard deviation over the last axis, then applies a gain $\gamma$ and (optionally) a shift $\beta$:

```math
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \qquad
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2, \qquad
y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

### BatchNorm2d

Normalizes each channel $c$ using a mean $\mu_c$ and variance $\sigma_c^2$, then applies a learned scale $\gamma_c$ and shift $\beta_c$:

```math
y_c = \gamma_c \, \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
```

**Eval:** $\mu_c$ and $\sigma_c^2$ are the stored running estimates $\hat\mu_c$ and $\hat\sigma_c^2$.

**Training:** they come from the current batch, averaged over the $N = B \cdot H \cdot W$ values in channel $c$:

```math
\mu_c = \frac{1}{N} \sum x_c, \qquad \sigma_c^2 = \frac{1}{N} \sum (x_c - \mu_c)^2
```

and the running estimates are updated with momentum $m$. The variance update uses the unbiased $\frac{N}{N-1}$ correction:

```math
\hat\mu_c \leftarrow (1 - m)\,\hat\mu_c + m\,\mu_c, \qquad
\hat\sigma_c^2 \leftarrow (1 - m)\,\hat\sigma_c^2 + m \, \frac{N}{N - 1} \, \sigma_c^2
```

## `convolution.py`

### Conv2d

Slides a $K \times K$ filter over the input. $\tilde{x}$ is the input padded with zeros ($p$ on every side, or a separate count per side), $s$ is the stride and $\delta$ the dilation (the gap between filter taps). For output channel $k$ at position $(i, j)$:

```math
y_{k,i,j} = b_k + \sum_{c \in \mathcal{G}(k)} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} W_{k,c,u,v} \; \tilde{x}_{c,\, s i + \delta u,\, s j + \delta v}
```

```math
H_\text{out} = \left\lfloor \frac{H + 2p - \delta (K - 1) - 1}{s} \right\rfloor + 1
```

and likewise for $W_\text{out}$. With $G$ groups, output channel $k$ sees only the $C_\text{in} / G$ input channels $\mathcal{G}(k)$ of its group. The code does this as one matrix multiply per group: each $K \times K$ patch is flattened into a column ("im2col") and multiplied by the flattened filters. Depthwise convolution ($G = C_\text{in} = C_\text{out}$) skips im2col and sums $K^2$ shifted copies of the input instead.

## `pooling.py`

### MaxPool2d

The largest value in each $K \times K$ window. The padding is $-\infty$ rather than zero, so padding never wins. The stride $s$ defaults to $K$:

```math
y_{c,i,j} = \max_{0 \le u, v < K} \; \tilde{x}_{c,\, s i + u,\, s j + v}
```

With `ceil_mode` the output size rounds up, so a final window may hang over the edge, unless that window would start entirely inside the padding.

### AdaptiveAvgPool2d

Averages over windows sized to hit a fixed output grid $H_o \times W_o$. Output row $i$ covers input rows

```math
\left\lfloor \frac{i \, H}{H_o} \right\rfloor \;\le\; r \;<\; \left\lceil \frac{(i + 1)\, H}{H_o} \right\rceil
```

and columns work the same way. $y_{c,i,j}$ is the mean of the input inside that window. When $H$ is not a multiple of $H_o$, neighbouring windows overlap.

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
