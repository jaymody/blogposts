---
title: "Speculative Sampling"
date: 2023-02-08
description: "A review of \"Accelerating Large Language Model Decoding with Speculative Sampling\" from Deepmind."
---
This post provides an overview, implementation, and time complexity analysis of DeepMind's paper [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318).

Code for this blog post can be found at [github.com/jaymody/speculative-samlping](https://github.com/jaymody/speculative-sampling).

[[toc]]

# Autoregressive Sampling
The standard way of generating text from a language model is with **autoregressive sampling**, here's the algorithm as defined in the paper:

![](https://i.imgur.com/YrLebkI.png)

In code:

```python
def autoregressive_sampling(x, model, N):
    n = len(x)
    T = len(x) + N

    while n < T:
        x = np.append(x, sample(model(x)[-1]))
        n += 1

    return x
```

Where:

* `model` is a language model (like GPT) accepts as input list of token ids of length `seq_len` and outputs a matrix of probabilities of shape `[seq_len, vocab_size]`.
* `N` is the number of tokens we we want to decode.

The time complexity of this algorithm is $O(N \cdot t_{\text{model}})$:

* $N$: The number of iterations of our while loop, which is just the number of tokens to decode $N$.
* $t_{\text{model}}$:  The time complexity of each iteration in the loop, which is just the time taken for a single forward pass of our model $t_{\text{model}}$.

# Speculative Sampling
In **speculative sampling**, we have two models:

1. A smaller, faster **draft model** (i.e. 7B Chinchilla GPT model)
2. A larger, slower **target model** (i.e. 70B Chinchilla GPT model)

Instead of decoding a single token at each iteration, speculative sampling decodes between 1 to $K + 1$ tokens per iteration:

1. The draft model decodes $K$ tokens autoregressively.
2. This new predicted sequence is passed as input to both the draft model and target models to get their respective probability outputs.
3. Using these probabilities, we determine how many of the predicted $K$ tokens we want to keep based on a **rejection criteria**. If a token is rejected, we resample it using a combination of the two distributions and don't accept any more tokens.
4. If all $K$ tokens were accepted, we sample an additional final token.

In short, the draft model _speculates_ what the output is $K$ steps into the future. The target model determines how many of those tokens we should accept. If our draft model is able to achieve a high enough acceptance rate and is sufficiently faster than the target model, then speculative sampling will yield a speedup.

You can imagine speculative sampling will work particularly well on common sequences of words. For example, the phrase "The apple doesn't fall far from the tree" is a common idiom in English. Given just "The apple doesn't fall", autoregressive decoding would require 4 forward passes of the target model, one for each word. In speculative sampling, with $K=4$, the draft model would predict "far from the tree" (since it is a common phrase), and the target model just has to do a single forward pass to verify that this is correct, saving time. Of course this won't occur every time, sometimes none of the $K$ predictions are accepted, sometimes only some of them but not all of them are accepted. 

Here's the full algorithm as defined in the paper:

![](https://i.imgur.com/rhR3U46.png)

In code:

```python
def speculative_sampling(x, draft_model, target_model, N, K):
    # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
    # we have to add an extra -1 term when indexing using n, T, or t
    n = len(x)
    T = len(x) + N

    while n < T:
        # Step 1: auto-regressive decode K tokens from draft model
        x_draft = x
        for _ in range(K):
            x_draft = np.append(x_draft, sample(draft_model(x_draft)[-1]))

        # Step 2: full draft and target model forward passes on x_draft
        p = draft_model(x_draft)
        q = target_model(x_draft)

        # Step 3: append draft tokens based on rejection criterion and resample
        # a token on rejection
        keep_n = 0
        for _ in range(K):
            r = np.random.random()
            i = n - 1
            j = x_draft[i + 1]
            if r < min(1, q[i][j] / p[i][j]):  # accepted
                x = np.append(x, j)
                n += 1
                keep_n += 1
            else:  # rejected
                x = np.append(x, sample(max_fn(q[i] - p[i])))  # resample
                n += 1
                break

        # Step 4: if all draft tokens were accepted, sample a final token
        if keep_n == K:
            x = np.append(x, sample(q[-1]))
            n += 1

        # just keeping my sanity
        assert n == len(x), f"{n} {len(x)}"

    return x
```

The time complexity for this algorithm is $O(\frac{N}{r(K + 1)} \cdot (t_{\text{draft}}(K + 1) + t_{\text{target}}))$.

* $\frac{N}{r(K+1)}$: The number of iterations in our while loop. This works out to the number of tokens we want to decode $N$ divided by the average number of tokens that get decoded per iteration $r(K + 1)$. The paper doesn't directly report the average number of tokens that get decoded per iteration, instead they provide the acceptance rate $r$ (which is the average number of tokens decoded per iteration divided by $K + 1$)[^acceptance]. As such, we can recover the average number of tokens decoded simply by multiplying $r$ by $K + 1$.
* $t_{\text{draft}}(K + 1) + t_{\text{target}}$: The time complexity for each iteration in the loop. The $t_{\text{target}}$ term is for the single forward pass we do for the target model. The term $t_{\text{draft}}(K + 1)$  is for all the forward passes of the draft model, for which there are $K + 1$ ($K$ for step 1 and the $+1$ from step 2).

# Speedup Results
The paper reports the following speedups for their 70B Chinchilla model (using a specially trained 7B Chinchilla as the draft model):

![](https://i.imgur.com/3ZcmZfr.png)

You can see that there was no performance degradation and the decoding process is 2 times faster as compared to autoregressive decoding.

Let's compare these empirical speedup numbers to the theoretical speedup numbers, which we can calculate using our time complexity equations:

$$
\begin{align}
\text{speedup} & = \frac{\text{time complexity of autoregressive}}{\text{time complexity of speculative}} \\
& = \frac{N\cdot t_{\text{target}}}{\frac{N}{r(K + 1)} \cdot (t_{\text{draft}}(K + 1) + t_{\text{target}})}
& \\
& = \frac{r(K + 1) \cdot t_{\text{target}}}{t_{\text{draft}}(K + 1) + t_{\text{target}}}
\end{align}
$$

Using the numbers provided in the paper:

* $K = 4$
* $t_{\text{draft}} = 1.8\text{ms}$
* $t_{\text{target}} = 14.1\text{ms}$
* $r = 0.8$ for HumanEval and $r = 0.62$ for XSum (see figure 1 in the paper)

For HumanEval we get a theoretical speedup of **2.44**, while the paper reports an empirical speedup of **2.46**.

For XSum we get a theoretical speedup of **1.89**, while the paper reports an empirical speedup of **1.92**.

We can reproduce these results by [running our implementation with GPT-2 1.5B as our target model and GPT-2 124M as our draft model](https://github.com/jaymody/speculative-sampling)[^result]:

```python
python main.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --draft_model_size "124M" \
    --target_model_size "1558M" \
    --K 4 \
    --seed 123
```

Which gives a speedup of **2.38**[^performance]:

```text
Autoregressive Decode
---------------------
Time = 71.64s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T

Speculative Decode
------------------
Time = 30.11s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think for themselves. But it's not just computers that are capable of thinking for themselves.

In fact, the brain is a computer, and it's capable
```

[^acceptance]: The wording from the paper for $r$ is a bit misleading. The paper states that $r$ is "the average number of tokens **accepted** divided by $K + 1$". This gives the impression they are reporting the rate at which **just** the draft tokens are accepted (i.e. don't include the resampled and final sampled tokens). In actuality, $r$ is "the average number of tokens **decoded** divided by $K + 1$" meaning we also includes the resampled and final token. This would make sense since otherwise, they would have to divided $r$ by $K$ and not $K + 1$ when reporting $r$. I confirmed this with the authors of the paper.
[^result]: The implementation for GPT-2 used here is a very naive (i.e. doesn't include KV caching among many other things). That is to say, the speedup results here should be taken with a grain of salt, but still it serves as a good validation for speculative sampling.
[^performance]: Of course, I have not verified that there is no performance degradation, but qualitatively, the output seems about right.