---
title: "An Intuition for Attention"
date: 2022-10-22
description: "Deriving the equation for scaled dot product attention."
---

The transformer neural network architecture is the secret sauce behind LLMs (large language models) like GPT-3 and the models that power [cohere.ai](https://cohere.ai). The main feature of the transformer is a mechanism called [attention](https://en.wikipedia.org/wiki/Attention_%28machine_learning%29). While attention has many different forms, the attention mechanism proposed in the original [transformer paper]( https://arxiv.org/pdf/1706.03762.pdf) is defined as:
$$\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
This version of attention (known as scaled dot product attention), is still widely used in many LLM architectures. In this post, we'll build an intution for the above equation by deriving it from the ground up.

To start, let's take a look at the problem attention aims to solve, the key-value lookup.

## Key-Value Lookups
---
A key-value lookup involves three components:
1. A list of $n_k$ **keys**
2. A list of $n_k$ **values** (that map 1-to-1 with the keys, forming key-value pairs)
3. A **query**, for which we want to _match_ with the keys and get some value based on the match

You're probably familiar with this concept as a [hash table](https://en.wikipedia.org/wiki/Hash_table) or dictionary. For example in python, we can perform a key-value lookup with a dictionary:
```python
# keys = racket, ball, tree
# values = 10, 5, 2
# query = racket
kv_pairs = {
    "racket": 10,
    "ball": 5,
    "tree": 2,
}

# returns 10
kv_pairs["racket"]
```

More generally, if we don't know our keys and values ahead of time:

```python
def kv_lookup(query, keys, values):
    for key, value in zip(keys, values):
        if query == key:
            return value
    raise ValueError(f"query {query} not found in keys {keys}")

# returns 10
kv_lookup("racket", ["racket", "ball", "tree"], [10, 5, 2])
```

One problem with our above implementation and dictionaries is that they only allow us to perform look ups based on exact string matches.

What if we wanted to look up queries based on the _meaning_ of a word?

## Key-Value Lookups based on Meaning
---
Say we wanted to look up the word "tennis" based on it's meaning. How do we choose which key matches tennis?

It's obviosly not "tree", but both "racket" and "ball" seem appropriate matches (tennis is a racket sport that involves a ball). It's hard to choose one or the other, tennis feels more like a combination of racket and ball rather than a strict match for either.

So, let's not choose, instead we'll do exactly that, take a combination of racket and ball. We'll encode how much _influence_ each key-value pair has on our output as a percentage, and then take the weighted sum of the values. For example, let's say we choose 60% for racket, 40% for ball, and 0% for tree:

```python
kv_lookup_by_meaning("tennis") == 0.6*10 + 0.4*5 + 0.0*2 == 8
```

In other words, we are determining how much **attention** our query should be paying to each key-value pair (i.e. 60% of the attention should be on the word racket, 40% on the word ball, etc ...). As such, we call the percentages _attention scores_. Mathematically[^mathnote], we can define our output as:
$$
\sum_{i} \alpha_iv_i
$$
where $\alpha_i$ is our attention score for the $i$th kv pair and $v_i$ is the $i$th value. The attention scores must be decimal percentages, that is, they must sum to 1 and be between 0 and 1.[^constraint]

But where did we get these attention scores from? In our example, I just kind of chose them based on what I _felt_. While I think I did a pretty good job, this approach doesn't seem sustainable (unless you can find a way to clone me and fit me into computer memory).

Instead lets take a look at how word vectors may help solve our problem of determining attention scores.

## Word Vectors and Similarity
---
Words can be represented with vectors that are capable of encapsulaling a word's _meaning_. In the context of neural networks, these vectors usually come from some kind of learned embedding or latent representation (don't worry about what this means, just know word vectors exist and are capable of encapsulating meaning). For example, given the words "King", "Queen", "Man", and "Women" and their respective vector representations $\boldsymbol{v}_{\text{king}}, \boldsymbol{v}_{\text{queen}}, \boldsymbol{v}_{\text{man}}, \boldsymbol{v}_{\text{women}}$, we can imagine that:
$$\boldsymbol{v}_{\text{queen}} - \boldsymbol{v}_{\text{woman}} + \boldsymbol{v}_{\text{man}} \sim \boldsymbol{v}_{\text{king}}$$
That is, the vector for queen - woman + man should result in a vector that is "similar" to that of king. There are many ways to [measure the similarity between two vectors](https://towardsdatascience.com/9-distance-measures-in-data-science-918109d069fa), possibly the simplest way is to compute their dot product:
$$\boldsymbol{v} \cdot \boldsymbol{w} = \sum_{i}v_i w_i$$
[3blue1brown has a great video on the intution behind dot product](https://www.youtube.com/watch?v=LyGKycYT2v0), but for our purposes all we need to know is:
* If two vectors are pointing in the same direction, the dot product will be > 0 (i.e. similar)
* If they are pointing in opposing directions, the dot product will be < 0 (i.e. disimilar)
* If they are exactly perpendicular, the dot product will be 0 (i.e. neutral)

Using this information, we can define a simple heuristic to determine the similarity between two word vectors: The higher the dot product, the more similar the two words are.[^magnitude]

## Attention Scores using the Dot Product
---
If we treat our queries and keys as word vectors instead of strings ($\boldsymbol{q} = \boldsymbol{v}_{\text{tennis}}$ and $\boldsymbol{k} = [\boldsymbol{v}_{\text{racket}} \ \boldsymbol{v}_{\text{ball}} \ \boldsymbol{v}_{\text{tree}}]$, all with dimensionality $d_k$), we can compute their similarity as a dot product:
$$
x_i = \boldsymbol{q} \cdot \boldsymbol{k}_i
$$
If we put our dot products into a vector $\boldsymbol{x} = [x_1, x_2, \ldots, x_{n_k - 1}, x_{n_k}]$, we can compute it with:

$$
\boldsymbol{x} = \boldsymbol{q}{K}^T
$$
where $K$ is a row-wise matrix of our key vectors (i.e. our key vectors stacked ontop of eachother to form a $n_k$ by $d_k$ matrix). If you're having trouble understanding this, see [^matmul].

Recall that our attention scores need to be decimal percentages (between 0 and 1 and sum to 1). Our dot product values on the other hand, can be any real number (i.e. between $-\infty$ and $\infty$). To transform our dot product values to decimal percentages, we'll need to use the [softmax function](https://en.wikipedia.org/wiki/Softmax_function):
$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
The softmax functions takes a vector of real numbers and transforms it such that it sums to 1 and each value is between 0 and 1. Softmax is also [monotonic](https://en.wikipedia.org/wiki/Monotonic_function), meaning the order by value is preserved (i.e. the highest value number will stay the highest value number, and the lowest value number), so our heuristic of higher values meaning more similar still holds after applying softmax. Here's an example of softmax in action:

```python
import numpy as np

def softmax(x):
    # assumes x is a vector
    return np.exp(x) / np.sum(np.exp(x))

# returns [0.8648, 0.0058, 0.1294]
# notice:
# - each number is between 0 and 1
# - the numbers sum to 1
# - the corresponding values are still in order
#   (4.0 is still largest, -1.0 is still lowest)
# - higher get more "weight"
softmax(np.array([4.0, -1.0, 2.1])) 
```

So, we can obtain our attention scores by taking the softmax of the dot products between our query word vector and key word vectors:
$$
\alpha_i = \text{softmax}(\boldsymbol{x})_i = \text{softmax}(\boldsymbol{q}K^T)_i
$$
Going back to our original output equation $\sum_{i}\alpha_iv_i$ and plugging things in, we get [^values]:
$$
\begin{align}
\sum_{i}\alpha_iv_i
= & \sum_i \text{softmax}(\boldsymbol{x})_iv_i\\
= & \sum_i \text{softmax}(\boldsymbol{q}K^T)_iv_i\\
= &\ \text{softmax}(\boldsymbol{q}K^T)\boldsymbol{v}
\end{align}
$$
With that, we've pretty much come to a working definition for attention:
$$
\text{attention}(\boldsymbol{q}, K, \boldsymbol{v}) = \text{softmax}(\boldsymbol{q}K^T)\boldsymbol{v}
$$
In code:

```python
import numpy as np

def get_word_vector(word, d_k=512):
    """Hypothetical mapping that returns a word vector of size
    d_k for the given word. For demonstrative purposes, we initialize
    this vector randomly, but in practice this would come from learned
    embeddings or some kind of latent representation."""
    return np.random.normal(size=(d_k,))

def softmax(x):
    # assumes x is a vector
    return np.exp(x) / np.sum(np.exp(x))

def attention(q, K, v):
    # assumes q is a vector of shape (d_k)
    # assumes K is a matrix of shape (n_k, d_k)
    # assumes v is a vector of shape (n_k)
    return softmax(q @ K.T) @ V

def kv_lookup(query, keys, values):
    return attention(
        q = get_word_vector(query),
        K = np.array([get_word_vector(key) for key in keys]),
        v = values,
    )

# returns a float number
print(kv_lookup("tennis", ["racket", "ball", "tree"], [10, 5, 2]))
```

## Scaled Dot Product Attention
---
In principle, the attention equation we derived in the last section is complete. However, we'll need to make a couple of changes to match the version in [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

##### Values as Vectors
The values associated with each key need not be a singular number, they can be vectors of any size $d_v$. For example with $d_v = 4$, you might have:

```python
kv_pairs = {
    "racket": [0.9, 0.2, -0.5, 1.0]
    "ball": [1.2, 2.0, 0.1, 0.2]
    "tree": [-1.2, -2.0, 1.0, -0.2]
}
```

Our attention equation in this case doesn't change much, the only difference is instead of multipy our attention scores by a vector $v$ we multiply it by the row-wise matrix of our value vectors $V$ (similar to how we stacked our keys to form $K$):
$$
\text{attention}(\boldsymbol{q}, K, V) = \text{softmax}(\boldsymbol{q}K^T)V
$$
Of course, in this case instead of our output being a single number, it would be a vector of dimensionality $d_v$.

#### Scaling
The dot product between our query and keys can get really large in magnitude if $d_k$ is large. This has a couple of undesirable consequences. First, the output of softmax becomes more _extreme_ instead of being a smooth distribution of values. For example, `softmax([3, 2, 1]) = [0.665, 0.244, 0.090]`, but with larger values (say we multiply our inputs by 10) `softmax([30, 20, 10]) = [9.99954600e-01, 4.53978686e-05, 2.06106005e-09]`. Second, the gradients of a neural network that implements attention will become extremely small, which hurts the training process. As a solution, we scale our pre-softmax scores by $\frac{1}{\sqrt(d_k)}$:

$$
\text{attention}(\boldsymbol{q}, K, V) = \text{softmax}(\frac{\boldsymbol{q}K^T}{\sqrt{d_k}})V
$$

#### Multiple Queries
In practice, we often want to do perform multiple lookups for $n_q$ different queries rather than just a single query. Of course, we could always do this one at a time, plugging each query in the above equation. However, if we stack of query vectors row-wise as a matrix $Q$ (in the same way we did for $K$ and $V$), we can compute our output as a $n_q$ by $d_v$ matrix where row $i$ is the output vector for the attention on the $i$th query:
$$
\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
that is, $\text{attention}(Q, K, V)_i = \text{attention}(q_i, K, V)$. In a practical setting, this makes computation faster than if we ran attention for each query seperately (say, in a for loop).

#### Result
WIth that, we have our final equation for scaled dot product attention as it's written in the original paper:
$$
\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$In code:

```python
import numpy as np

def softmax(x):
    # assumes x is a matrix and we want to take the softmax along each row
    # (which is achieved using axis=-1)
    return np.exp(x) / np.sum(np.exp(x), axis=-1)

def attention(Q, K, V):
    # assumes Q is a matrix of shape (n_q, d_k)
    # assumes K is a matrix of shape (n_k, d_k)
    # assumes v is a matrix of shape (n_k, d_v)
    # output is a matrix of shape (n_q, d_v)
    d_k = K.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d_k)) @ V
```


[^mathnote]: We use the [standard math notation](https://www.deeplearningbook.org/contents/notation.html) defined in the [Deep Learning](https://www.deeplearningbook.org) book. The relevant bits for this post: Scalars are unbolded ($a$), vectors are bolded ($\boldsymbol{a}$), Matrices are uppercase ($A$)
[^magnitude]: You'll note that the magnitude of the vectors have an influence on the output of dot product. For example, given 3 vectors, $a=[1, 1, 1]$, $b=[1000, 0, 0]$, and $c=[2, 2, 2]$, our dot product heuristic would tell us that becuase $a \cdot b > a \cdot c$  that $a$ is more similar to $c$ than $a$ is to $b$. This doesn't seem right, since $b$ and $a$ are pointing in the exact same direction, while $c$ and $a$ are not. [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) accounts for this normalizing the vectors to unit vectors before taking the dot product, essentially ignoring the magnitudes and only caring about the direction. So why don't we take the cosine similarity? In deep learning settings, the magnitude of a vector might actually contain information we care about (and we shouldn't get rid of it). Also, if we regularize our networks properly, outlier examples like the above should not occur.
[^constraint]: In mathematicall notation, this constraint can be defined as: $\forall i \in n_k :0 \le\alpha_i \le1  \land \sum_i \alpha_i =1$
[^values]: In the last step, we pack our values into a vector $\boldsymbol{v} = [v_1, v_2, ..., v_{n_k -1}, v_{n_k}]$, which allows us to get rid of the summation notation in favour of a dot product.
[^matmul]: Basically, instead of computing each dot product seperately:
$$
\begin{align}
x_1 = & \ \boldsymbol{q} \cdot \boldsymbol{k}_1 = [2, 1, 3] \cdot [-1, 2, -1] = -3\\
x_2 = & \ \boldsymbol{q} \cdot \boldsymbol{k}_2 = [2, 1, 3] \cdot [1.5, 0, -1] = 0\\
x_3 = & \ \boldsymbol{q} \cdot \boldsymbol{k}_3 = [2, 1, 3] \cdot [4, -2, -1] = 3
\end{align}
$$
You compute it all at once:
$$
\begin{align}
\boldsymbol{x} & = \boldsymbol{q}{K}^T \\
& = \begin{bmatrix}2 & 1 & 3\end{bmatrix}\begin{bmatrix}-1 & 2 & -1\\1.5 & 0 & -1\\4 & -2 & -1\end{bmatrix}^T\\
& = \begin{bmatrix}2 & 1 & 3\end{bmatrix}\begin{bmatrix}-1 & 1.5 & 4\\2 & 0 & -2\\-1 & -1 & -1\end{bmatrix}\\
& = [-3, 0, 3]\\
& = [x_1, x_2, x_3]
\end{align}
$$