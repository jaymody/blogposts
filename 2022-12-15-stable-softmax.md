---
title: Numerically Stable Softmax and Cross Entropy
date: 2022-12-15
description: Tricks to make softmax and cross entropy calculations numerically stable.
---
In this post, we'll take a look at softmax and cross entropy loss. We'll see that naive implementations result in numerically instability, and then we'll derive implementations that are numerically stable.

## Symbols
---
* $x$: Input vector of dimensionality $d$.
* $y$: Correct class, an integer on the range $y \in [1\ldots K]$.
* $\hat{y}$: Raw outputs (i.e. logits) of our neural network, vector of dimensionality $K$.
* We use $\log$ to denote the natural logarithm.

## Softmax
---
The softmax function is defined as:
$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

In code:
```python
def softmax(x):
    # assumes x is a vector
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1.2, 2, -4, 0.0])
softmax(x)
# outputs: [0.28310553, 0.63006295, 0.00156177, 0.08526975]
```

For large inputs we start seeing some numerical instability:
```python
x = np.array([1.2, 2000, -4000, 0.0])
softmax(x)
# outputs: [0., nan, 0.,  0.]
```

Floating point numbers aren't magic, they have limits:
```python
np.finfo(np.float64).max
# 1.7976931348623157e+308, largest positive number

np.finfo(np.float64).tiny
# 2.2250738585072014e-308, smallest positive number at full precision

np.finfo(np.float64).smallest_subnormal
# 5e-324, smallest positive number
```

When we go beyond these limits, we start seeing funky behaviour:
```python
np.finfo(np.float64).max * 2 
# inf, overflow error

np.inf - np.inf
# nan, not a number error

np.finfo(np.float64).smallest_subnormal / 2
# 0.0, underflow error
```

Looking back at our softmax example that resulted in `[0., nan, 0.,  0.]`, we can see that the overflow of `np.exp(2000) = np.inf` is causing the `nan`, since we end up with `np.inf / np.inf = nan`. If we want to avoid `nans`, we need to avoid `infs`. To avoid `infs`, we need to avoid overflows. To avoid overflows, we need to prevent our numbers from growing too large.

Underflows on the other hand don't seem quite as detrimental. Worst case scenario, we get a result `0` and lose all precision (i.e. `np.exp(-4000) = 0)`. While this is not ideal, this is a lot better than running into `inf` and `nan`.

Given the relative stability of floating point underflows vs overflows, how can we fix softmax?

Let's revisit our softmax equation and apply some tricks:
$$
\begin{align}
\text{softmax}(x)_i
&= \frac{e^{x_i}}{\sum_j e^{x_j}} \\
&= 1\cdot \frac{e^{x_i}}{\sum_j e^{x_j}} \\
&= \frac{C}{C}\frac{e^{x_i}}{\sum_j e^{x_j}} \\
&= \frac{Ce^{x_i}}{\sum_j Ce^{x_j}} \\
&= \frac{e^{x_i + \log C}}{\sum_j e^{x_j + \log C}} \\
\end{align}
$$
Here, we're taking advantage of the rule $a\cdot b^x = b^{x + \log_b a}$. As a result, we are given the ability to offset our inputs by any constant we choose. So, if we want to prevent large numbers, we can simply set $\log C = -\max(x)$:
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

This version is numerically stable:

* All exponentiated values will be between 0 and 1: $0 \leq e^{x_i - \max(x)} \leq 1$
    * This prevents overflow errors (but we are still prone to underflows)
* At least one of the exponentiated values is 1: $e^{x_i - \max(x)} = e^{ \max(x)- \max(x)} = e^0 = 1$
    * i.e. at least one value is guarenteed not to underflow
    * Our denominator will always be $>= 1$, preventing division by zero errors.
    * We have at least one non-zero numerator, so softmax can't result in a zero vector

In code:
```python
def softmax(x):
    # assumes x is a vector
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1.2, 2, -4, 0])
softmax(x)
# outputs: [0.28310553, 0.63006295, 0.00156177, 0.08526975]

# works for large numbers!!!
x = np.array([1.2, 2, -4, 0]) * 1000
softmax(x)
# outputs: [0., 1., 0., 0.]
```

## Cross Entropy and Log Softmax
---
The cross entropy between two probability distributions is defined as.
$$
H(p, q) = -\sum_i p_i\log(q_i)
$$
where $p$ and $q$ are our probability distributions represented as probability vectors (that is $p_i$ and $q_i$are the probabilities of event $i$ occuring for $p$ and $q$ respectively).

Roughly speaking, cross entropy measures the similarity between two probability distributions, so in the context of machine learning, we use cross entropy as a loss function where:

* $q$ is our predicted probabilities vector (i.e. the softmax of our network outputs): $q = \text{softmax}(\hat{y})$
* $p$  is a one-hot encoded vector of our label, that is a probability vector that assigns 100% probability to the position $y$ (our label for the correct class): $p_i = \begin{cases} 1 & i = y \\ 0 & i \neq y \end{cases}$

In this setup, our cross entropy simplifies to:
$$
\begin{align}
H(p, q)
&= -\sum_i p_i\log(q_i) \\
&= -p_y\cdot\log(q_y) -\sum_{i \neq y} p_i\log(q_i) \\
&= -1\cdot\log(q_y) -\sum_{i \neq y} 0\cdot\log(q_i) \\
&= -\log(q_y) - 0 \sum_{i \neq y} \log(q_i) \\
&= -\log(q_y) \\
&= -\log(\text{softmax}(\hat{y})_y)
\end{align}
$$

In code:

```python
def cross_entropy(y_hat, y_true):
    # assume y_hat is a vector and y_true is an integer
    return -np.log(softmax(y_hat)[y_true])

cross_entropy(
    y_hat=np.random.normal(size=(10)),
    y_true=3,
)
# 2.580982279204241
```

For large numbers in `y_hat`, we start seeing `inf`:

```python
cross_entropy(
    y_hat = np.array([-1000, 1000]),
    y_true = 0,
)
# inf
```

In this case `softmax(y_hat) = [0, 1]`, with `y_true = 0`, we get `-log(0) = inf`. So to make `log(softmax(x))` stable, we need to prevent `log(0)`. Let's see if we can employ some math tricks to make this happen:
$$
\begin{align}
\log(\text{softmax}(x)_i)
& = \log(\frac{e^{x_i - \min(x)}}{\sum_j e^{x_j - \min(x)}}) \\
&= \log(e^{x_i - \min(x)}) - \log(\sum_j e^{x_j - \min(x)}) \\
&= (x_i - \min(x))\log(e) - \log(\sum_j e^{x_j - \min(x)}) \\
&= (x_i - \min(x))\cdot 1 - \log(\sum_j e^{x_j - \min(x)}) \\
&= x_i - \min(x) - \log(\sum_j e^{x_j - \min(x)}) \\
\end{align}
$$
This new equation guarentees that the sum inside the log will always be $\geq 1$, so we no longer need to worry about `log(0)` errors.

In code:

```python
def log_softmax(x):
    # assumes x is a vector
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))

def cross_entropy(y_hat, y_true):
    return -log_softmax(y_hat)[y_true]

cross_entropy(
    y_hat=np.random.normal(size=(10)),
    y_true=3,
)
# 2.580982279204241

# works for large inputs!!!!
cross_entropy(
    y_hat = np.array([-1000, 1000]),
    y_true = 3,
)
# 705.3963550098291
```