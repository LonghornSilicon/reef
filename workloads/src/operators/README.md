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

Position-derived tensors that are added to embeddings or to attention scores.

### SinePositionEmbedding2d

DETR's encoding of pixel coordinates. From a pixel mask $M$ (1 where the pixel is real), the row and column of each pixel are its cumulative counts, normalised to $[0, 2\pi]$ by the last count on that axis:

```math
\bar{y} = \frac{\mathrm{cumsum}_H(M) - o}{\mathrm{cumsum}_H(M)_{\text{last}} + \epsilon} \cdot 2\pi, \qquad
\bar{x} = \frac{\mathrm{cumsum}_W(M) - o}{\mathrm{cumsum}_W(M)_{\text{last}} + \epsilon} \cdot 2\pi
```

with offset $o = 0$ for DETR and $o = \tfrac12$ for Deformable DETR. Each coordinate is then spread over $n$ features at geometric frequencies, sine on the even ones and cosine on the odd:

```math
t_i = T^{2 \lfloor i/2 \rfloor / n}, \qquad
\mathrm{PE}(\bar{x})_{2k} = \sin\frac{\bar{x}}{t_{2k}}, \quad
\mathrm{PE}(\bar{x})_{2k+1} = \cos\frac{\bar{x}}{t_{2k+1}}
```

The output channels are $[\mathrm{PE}(\bar{y}), \mathrm{PE}(\bar{x})]$.

### SinCosPositionEmbedding2d

MAE's fixed table over an integer patch grid. With $d/4$ frequencies $\omega_k = T^{-4k/d}$ and patch row $r$, column $c$:

```math
y = [\sin(r\,\omega),\ \cos(r\,\omega),\ \sin(c\,\omega),\ \cos(c\,\omega)]
```

### RelativePositionBias

T5's per-head bias, looked up by the bucket of the offset $j - i$ from query $i$ to key $j$. The first $n/2$ buckets are exact offsets; beyond that they widen logarithmically until offset $D$ (max distance):

```math
\mathrm{bucket}(r) =
\begin{cases}
r & r < n/2 \\
n/2 + \left\lfloor \dfrac{\log(r / (n/2))}{\log(D / (n/2))} \,(n - n/2) \right\rfloor & \text{otherwise}
\end{cases}
```

Bidirectional (encoder) use splits the buckets by sign and buckets $|r|$; causal (decoder) use buckets $\max(-r, 0)$. The bias for head $h$ is $E_{\mathrm{bucket}(j-i),\,h}$, added to the attention scores before the softmax.

## `activation.py`

All apply to each element on its own.

### ReLU

```math
\mathrm{ReLU}(x) = \max(0, x)
```

### ReLU6

```math
\mathrm{ReLU6}(x) = \min(\max(0, x), 6)
```

### SiLU

```math
\mathrm{SiLU}(x) = x \cdot \sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
```

### Sigmoid

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

### Tanh

```math
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
```

### GELU (tanh approximation)

```math
\mathrm{GELU}(x) = \frac{x}{2} \left( 1 + \tanh\left( \sqrt{2/\pi} \, \left( x + 0.044715 \, x^3 \right) \right) \right)
```

### ErfGELU

The exact form, $x$ times the Gaussian CDF:

```math
\mathrm{GELU}(x) = \frac{x}{2} \left( 1 + \mathrm{erf}\left( \frac{x}{\sqrt{2}} \right) \right)
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

### GroupNorm

Splits the $C$ channels into $G$ groups and normalises each group over its channels and every spatial position of one sample, then applies a per-channel $\gamma_c$, $\beta_c$. With group $g$ holding $N = (C/G) \cdot H \cdot W$ values:

```math
\mu_g = \frac{1}{N} \sum x_g, \qquad \sigma_g^2 = \frac{1}{N} \sum (x_g - \mu_g)^2, \qquad
y_c = \gamma_c \, \frac{x_c - \mu_{g(c)}}{\sqrt{\sigma_{g(c)}^2 + \epsilon}} + \beta_c
```

$G = C$ is InstanceNorm; $G = 1$ normalises the whole sample.

### L2Norm

Scales each vector along an axis to unit length. The clamp keeps a zero vector from dividing by zero:

```math
y = \frac{x}{\max(\|x\|_2, \epsilon)}
```

### GemmaRMSNorm

Same as RMSNorm, but the gain is stored as an offset from 1, so $w$ starts at zero:

```math
y = (1 + w) \odot \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}}
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

### FrozenBatchNorm2d

BatchNorm2d in eval mode with every term fixed, folded into one scale and shift per channel, as torchvision's detection backbones store it:

```math
s_c = \frac{\gamma_c}{\sqrt{\hat\sigma_c^2 + \epsilon}}, \qquad
y_c = s_c \, x_c + (\beta_c - \hat\mu_c \, s_c)
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

### ConvTranspose2d

The adjoint of a stride-$s$ convolution. The input is spread out by inserting $s - 1$ zeros between neighbours, padded by $K - 1 - p$, and correlated at stride 1 with the filter flipped in both spatial axes and its channel axes swapped:

```math
H_\text{out} = (H - 1)\, s - 2p + K
```

## `pooling.py`

### MaxPool2d

The largest value in each $K \times K$ window. The padding is $-\infty$ rather than zero, so padding never wins. The stride $s$ defaults to $K$:

```math
y_{c,i,j} = \max_{0 \le u, v < K} \; \tilde{x}_{c,\, s i + u,\, s j + v}
```

With `ceil_mode` the output size rounds up, so a final window may hang over the edge, unless that window would start entirely inside the padding.

### AvgPool2d

The mean of each $K \times K$ window. The divisor $N_{i,j}$ counts the real pixels in the window, plus the explicit padding when `count_include_pad`, and never the `ceil_mode` overhang:

```math
y_{c,i,j} = \frac{1}{N_{i,j}} \sum_{0 \le u, v < K} \tilde{x}_{c,\, s i + u,\, s j + v}
```

### AdaptiveAvgPool2d

Averages over windows sized to hit a fixed output grid $H_o \times W_o$. Output row $i$ covers input rows

```math
\left\lfloor \frac{i \, H}{H_o} \right\rfloor \;\le\; r \;<\; \left\lceil \frac{(i + 1)\, H}{H_o} \right\rceil
```

and columns work the same way. $y_{c,i,j}$ is the mean of the input inside that window. When $H$ is not a multiple of $H_o$, neighbouring windows overlap.

## `interpolation.py`

### Interpolate

Resizes the last two axes separably: one weight matrix per axis, $R \in \mathbb{R}^{H_o \times H}$ and $Q \in \mathbb{R}^{W_o \times W}$, so that

```math
y = R \, x \, Q^\top
```

Output index $i$ maps to source coordinate $u$ with ratio $\rho = H / H_o$ (or $1 / \text{scale factor}$ when one is given):

```math
u = (i + \tfrac12)\, \rho - \tfrac12 \quad\text{(default)}, \qquad
u = i \, \frac{H - 1}{H_o - 1} \quad\text{(align corners)}
```

Row $i$ of $R$ then holds the tap weights around $u$: one weight of 1 at $\lfloor i \rho \rfloor$ for `nearest`; $(1 - t, t)$ at $\lfloor u \rfloor, \lfloor u \rfloor + 1$ with $t = u - \lfloor u \rfloor$ for `bilinear` (after clamping $u \ge 0$); and Keys' cubic kernel with $a = -0.75$ on the four taps $\lfloor u \rfloor - 1 \ldots \lfloor u \rfloor + 2$ for `bicubic`. Taps past the edge clamp to the edge, so their weights pile up on the border pixel.

### GridSample

Bilinear sampling at arbitrary normalised coordinates $(g_x, g_y) \in [-1, 1]^2$. The coordinate unnormalises to pixels

```math
u = \frac{(g_x + 1)\, W - 1}{2} \quad\text{(default)}, \qquad u = \frac{g_x + 1}{2}\,(W - 1) \quad\text{(align corners)}
```

and the output is the weighted sum of the four surrounding pixels, where any pixel outside the input contributes zero:

```math
y = \sum_{a, b \in \{0, 1\}} w_a^x \, w_b^y \; x_{\lfloor v \rfloor + b,\; \lfloor u \rfloor + a}, \qquad
w_1^x = u - \lfloor u \rfloor,\ w_0^x = 1 - w_1^x
```

## `rotary.py`

### RotaryEmbedding (RoPE)

Encodes a token's position $n$ by rotating pairs of features through position-dependent angles. For rotary width $r$ (the head width by default; Pythia and Phi rotate only a leading slice), base $\theta$ and scaling factor $s$, there is one frequency per pair $k = 0, \dots, r/2 - 1$:

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

## `attention.py`

### GroupedQueryAttention

Inputs are queries $Q$ of shape $(B, h, L_q, d)$ and keys and values $K, V$ of shape $(B, h_{kv}, L_k, d)$. Each of the $h_{kv}$ key/value heads is shared by $h / h_{kv}$ query heads, so $K$ and $V$ are first repeated along the head axis to $h$ heads.

**1. Scores.** How much each query matches each key. The scale is $1/\sqrt{d}$ by default (Gemma 3 uses its own, T5 uses 1). An optional additive bias $A$ (relative position bias, ALiBi, Swin's table) is added here:

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

An optional boolean mask (true where disallowed) is OR-ed in; padding masks and bidirectional windows arrive this way. Disallowed scores are set to $-\infty$, which the code approximates with the dtype's most negative value, so a fully masked row softmaxes to uniform rather than NaN.

**3. Weights and output.** Each query's scores become probabilities, which mix the values:

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(S) \, V
```

### MultiScaleDeformableAttention

Deformable DETR's attention: instead of scoring every key, each query samples a few points near a reference location on each feature level. For query $q$, head $h$, level $l$ and point $p$, a linear layer predicts an offset $\Delta_{hlp}$ and an unnormalised weight $a_{hlp}$ (softmaxed over all $L \cdot P$ points of the head). The sampling location on level $l$ of size $(H_l, W_l)$ is

```math
\phi_{hlp} = r_l + \frac{\Delta_{hlp}}{(W_l, H_l)} \quad\text{(point references)}, \qquad
\phi_{hlp} = r_l^{xy} + \frac{\Delta_{hlp}}{P} \cdot \frac{r_l^{wh}}{2} \quad\text{(box references)}
```

and the output of head $h$ is the weighted sum of bilinearly sampled projected values:

```math
y_h = \sum_{l=1}^{L} \sum_{p=1}^{P} a_{hlp} \; \mathrm{GridSample}\big(W_v V_l,\ 2\phi_{hlp} - 1\big)
```

followed by an output projection over the concatenated heads.

## `detection.py`

Boxes are $(x_1, y_1, x_2, y_2)$ unless stated. A box has centre $(c_x, c_y)$, width $w = x_2 - x_1$ and height $h = y_2 - y_1$.

### box_iou

Intersection over union of every pair from two sets:

```math
\mathrm{IoU}(a, b) = \frac{|a \cap b|}{|a| + |b| - |a \cap b|}
```

### BoxCoder

Faster R-CNN's regression targets relative to an anchor $(c^a_x, c^a_y, w^a, h^a)$, with per-coordinate weights $(\omega_x, \omega_y, \omega_w, \omega_h)$:

```math
\delta = \left( \omega_x \frac{c_x - c^a_x}{w^a},\ \omega_y \frac{c_y - c^a_y}{h^a},\ \omega_w \log \frac{w}{w^a},\ \omega_h \log \frac{h}{h^a} \right)
```

Decoding inverts this; the log ratios are clamped at $\log(1000/16)$ so $\exp$ cannot overflow (optional, because EfficientDet decodes without a clamp).

### AnchorDecode

YOLOv5's decode from raw logits $t$ in grid cell $c$ with anchor $(a_w, a_h)$ and stride $s$:

```math
(x, y) = \big(2\sigma(t_{xy}) - \tfrac12 + c\big)\, s, \qquad (w, h) = \big(2\sigma(t_{wh})\big)^2 \odot (a_w, a_h)
```

### DistanceToBox

YOLOv8's anchor-free decode from distances $(l, t, r, b)$ to the four sides around an anchor point $(p_x, p_y)$:

```math
(x_1, y_1) = (p_x - l,\ p_y - t), \qquad (x_2, y_2) = (p_x + r,\ p_y + b)
```

optionally returned as centre and size.

### DFL

Each side distance is predicted as a distribution over $n$ integer bins. The decoded distance is the expected bin index, computed as a softmax followed by a fixed $1 \times 1$ convolution whose weights are $0, 1, \dots, n - 1$:

```math
d = \sum_{k=0}^{n-1} k \; \mathrm{softmax}(z)_k
```

### NMS

Greedy suppression. Boxes are visited in decreasing score order; each kept box removes every later box overlapping it by more than the IoU threshold. Class-aware suppression shifts each class's boxes by $\text{class} \cdot (\max \text{coordinate} + 1)$ so boxes of different classes can never overlap.

### RoIAlign

Pools a region $(x_1, y_1, x_2, y_2)$ of a feature map (scaled by the spatial scale $\lambda$) onto an $H_o \times W_o$ grid. Bin $(i, j)$ spans $h_b = \lambda h / H_o$ by $w_b = \lambda w / W_o$ and is sampled at an $n \times n$ grid of points

```math
\Big( \lambda x_1 + w_b \big(j + \tfrac{k + 1/2}{n}\big),\ \lambda y_1 + h_b \big(i + \tfrac{m + 1/2}{n}\big) \Big), \qquad 0 \le k, m < n
```

each bilinearly interpolated from the feature map and averaged. `aligned` shifts every coordinate by $-\tfrac12$ (pixel centres); otherwise the region is at least one pixel wide.
