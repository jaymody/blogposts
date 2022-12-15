---
title: Numerically Stable Softmax and Cross Entropy
date: 2022-12-15
description: Tricks to make softmax and cross entropy calculations numerically stable.
---
In this post, we'll look at some common tricks for overcoming the numerical instability of softmax and cross entropy loss.

## Symbols
---
* $\vec{x}$: Input vector of dimensionality $d$.
* $y$: Correct class (scalar), $y \in [1\ldots K]$.
* $f(x) = \hat{y}$: Output vector of our neural network, has dimensionality $K$.
* $\log = \log_e = \ln$: We use $\log$ to denote the natural logarithm.

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
```

An example of softmax in action:
```python
x = np.array([1.2, 2, -4, 0])
softmax(x)
# outputs: [0.28310553, 0.63006295, 0.00156177, 0.08526975]
```

However, for large inputs we start seeing some numerical instability:
```python
x = np.array([1.2, 2, -4, 0]) * 1000
softmax(x)
# outputs: [nan, nan, 0.,  0.]
```

Floating point numbers aren't magic, they have limits. For example, `np.float64` has the following limits:
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
# inf, overflow

np.inf - np.inf
# nan, not a number error

np.finfo(np.float64).smallest_subnormal / 2
# 0.0, underflow
```

Looking back at our softmax example, `softmax(np.array([1200, 2000, -4000, 0])) = [nan nan  0.  0.]`, we can see the nans are being cause by `np.inf / np.inf`. To avoid `nans`, we need to avoid `infs`. To avoid `infs`, we need to avoid overflows. To avoid overflows, we need to prevent our numbers from growing too large.

Underflows on the other hand don't seem quite as detrimental. Worst case scenario, we lose all precision and the resulting value becomes `0`, which is a lot better than `inf` and `nan`.

Given the relative stability of floating point underflows vs overflow, how can we fix softmax?

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
Here, we're taking advantage of the  rule $a\cdot b^x = b^{x + \log_b a}$. The advantage of this version of softmax is that we can offset our inputs by any constant we choose. So if we want to prevent large numbers, we can simply set $\log C = -\max(x)$, giving us a numerically stable softmax:
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

This has the cool property that no matter how large our numbers are:

* Exponentiated values will always be less than or equal to $e^0 = 1$, preventing overflow.
* The denominator will always be $>= 1$, preventing division by zero errors.

In code:
```python
def softmax(x):
    # assumes x is a vector
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

And it works! Even for large inputs!
```python
x = np.array([1.2, 2, -4, 0])
softmax(x)
# outputs: [0.28310553, 0.63006295, 0.00156177, 0.08526975]

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
where $p$ and $q$ are probability vectors representing the two distributions, that is $p_i$ and $q_i$are the probabilities of event $i$ occuring for $p$ and $q$ respectively.

In the context of machine learning, we use cross entropy as a loss function where:
* $q$ is our predicted probabilities vector (i.e. the softmax of our network outputs): $q = \text{softmax}(f(x))$
* $p$  is a one-hot encoded vector of our label, that is a vector of zeros with a single 1 at the position of the correct class index: $p_i = \begin{cases} 1 & i = y \\ 0 & i \neq y \end{cases}$

In this setup, our cross entropy simplifies to:
$$
\begin{align}
H(p, q)
&= -\sum_i p_i\log(q_i) \\
&= -p_y\cdot\log(q_y) -\sum_{i \neq y} p_i\log(q_i) \\
&= -1\cdot\log(q_y) -\sum_{i \neq y} 0\cdot\log(q_i) \\
&= -\log(q_y) - 0 \sum_{i \neq y} \log(q_i) \\
&= -\log(q_y)
\end{align}
$$
Subbing in for $q_y$ we get:
$$
H(p, q) = -\log(\text{softmax}(\hat{y})_y)
$$
where $\hat{y} = f(x)$ is just the raw output vector of the network.

In code:

```python
def cross_entropy(y_hat, y_true):
    # assume y_hat is a vector and y_true is an integer
    return -np.log(softmax(y_hat)[y_true])
```

Here it is in action:

```python
# dummy neural network outputs and label
cross_entropy(
    y_hat=np.random.normal(size=(10)),
    y_true=3,
)
# 2.580982279204241
```

But if for some reason `y_hat` contains large numbers, we start getting `inf`:
```python
# dummy neural network outputs and label
cross_entropy(
    y_hat = np.random.normal(size=(10)) * 1000,
    y_true = 3,
)
# inf
```

Well that's no good. Furthermore, if there's a 0 after softmax at the index of the correct class, we get `inf` due to `log(0)`:
```python
cross_entropy(
    y_hat = np.array([-1000000, 1000000]),
    y_true = 0,
)
# inf
```

As such, a naive application of $\log(\text{softmax}(x))$ is numerically unstable. To make it more stable, we can employ some math tricks:
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
Not only is this equation much more numerically stable, but it also has the nice property that the sum inside the log will always be $\geq 1$, meaning we don't have to worry about `log(0)` errors.

In code:

```python
def log_softmax(x):
    # assumes x is a vector
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))

def cross_entropy(y_hat, y_true):
    return -log_softmax(y_hat)[y_true]
```

And it works! Even for large inputs!
```python
cross_entropy(
    y_hat=np.random.normal(size=(10)),
    y_true=3,
)
# 2.580982279204241

cross_entropy(
    y_hat = np.random.normal(size=(10)) * 1000,
    y_true = 3,
)
# 705.3963550098291
```

