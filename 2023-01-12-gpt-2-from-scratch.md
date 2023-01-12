---
title: "gpt-2-from-scratch"
date: 2023-01-12
description: "Building a GPT model from scratch using only `jax`"
---
In this post, we'll build a [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) model from scratch using only pure functional [`jax`](https://github.com/google/jax) and test our implementation using the official GPT-2 model checkpoints released by OpenAI. I've broken things down into 4 sections:

* Input / Output: We start by working our way up to defining the Input / Output of a GPT and how it enables us to generate entire sentences. By the end of this section, you'll be able to understand the sentence "GPT-3 is just a large transformer based auto regressive language model that was trained on the internet via self-supervised learning".
* GPT Architecture: This is the actual meat of the post where we build a GPT implementation that exactly matches GPT-2 using only `jax` in a functional style.
* Testing our Implementation: Since our implementation matches that of GPT-2, we can test it by loading the official GPT-2 model weights from OpenAI and verify it gives correct outputs.
* Discussion: A random assortment of discussion questions.

This post assumes some experience with training neural networks. At minumum, you should understand the MLP architecture in and out, maybe having coded one from scratch.

All the code for this post can be found at [github.com/jaymody/gpt-jax](https://github.com/jaymody/gpt-jax).

[[toc]]

## Input / Output
### Language Model
The fundamental task that a GPT performs is language modeling, the task of predicting the next logical word in a sequence. For example, given "Not all heroes wear", the next logical word in the sequence is "capes".

With that, we can define the interface for our GPT:

```python
def gpt(inputs: str) -> str:
    next_word = # beep boop, get next word
    return next_word

next_word = gpt(text="not all heroes wear") # "capes"
```

### Auto-Regressive
To predict entire sentences, not just a single word, we can call our `gpt` function iteratively. At each iteration we append the predicted word to our input for the next `gpt` function call:

```python
def generate(inputs: str, n_words_to_generate: int) -> list[str]:
    for _ in range(n_words_to_generate):
        next_word = gpt(inputs):
        inputs += " " + next_word
    return inputs

generate(["not", "all"], params, 3)  # "not all heroes wear capes"
```

This process of predicting a future value (regression), and adding it back into the input (auto) is why you might see GPTs be referred to as "auto-regressive language models".

### Neural Network
Of course, GPTs are neural networks, and neural networks have parameters:
```python
def gpt(inputs: str, params) -> str:
    next_word = # beep boop, do neural net stuff to get next word prediction
    return next_word

next_word = gpt(text="not all heroes wear", params) # "capes"
```
For now, don't worry about what exactly the `params` variable looks like or where we get them from, just know it exists and needs to be passed to the `gpt` function.

### Tokenization
Neural networks work with numbers, not strings. So we need a way of representing a string as numbers, and be able to convert those numbers back into a string. We can do this my mapping individual words in our strings to a number using some kind of lookup table for words:

```python
class SpaceTokenizer:
    def __init__(self, vocab_list: list[str]):
        self.word_to_idx = {word: i for i, word in enumerate(vocab_list)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab_list)}
    
    def encode(self, text: str) -> list[int]:
        words = text.split(" ")
        ids = [self.word_to_idx[word] for word in words]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        words = [self.idx_to_word[i] for i in ids]
        text = " ".join(words)
        return text

tokenizer = SpaceTokenizer(["the", "a", "capes", "heroes", "not", "all", "wear"])
tokenizer.encode("not all heroes wear capes") # [4, 5, 3, 6, 2]
tokenizer.decode([4, 5, 3, 6, 2]) # "not all heros wear capes"
```

Here, we create a mapping from words to integers (and vice versa) using the index of the word in the vocabulary list as the integer. To break our string into words, we just split by a space (and vice versa to convert the indices back into a string). This process of breaking down a string into smaller parts is called tokenization.

In practice, splitting by spaces is very bad tokenization method. For example, what happens when we encounter a word not in our vocabulary? What happens if two words are seperated by multiple spaces, do we just get a token that represents a blank space? What about hypenated words? For example, given the word "check-in", does this mean we have to maintain seperate entries for "check", "in", and "check-in" in our vocab? That seems kind of wasteful.

There are lots of better tokenizers out there. GPT-2 uses Byte-Pair Encoding, the implementation for which can be found [here](). Assuming the necessary `vocab.bpe` and `encoder.json` files are downloaded and put in the correct location, we can test out this tokenizer:

```python
from encoder import get_encoder

tokenizer = get_encoder(model_name, models_dir)
ids = tokenizer.encode("not all heroes wear capes") # [1662, 477, 10281, 5806, 1451, 274]
tokens = [tokenizer.decoder[i] for i in ids] # ['not', 'Ġall', 'Ġheroes', 'Ġwear', 'Ġcap', 'es']
text = tokenizer.decode(ids) # "not all heroes wear capes"
```

Notice a couple wierd things:

1) Spaces are included in the words, and they are represented with the character `Ġ`.
2) "capes" was broken down into two words, "cap" and "es".

Importantly, this means our sequence of "words" may no longer actually be actual "words", so we refer to them as "tokens" instead of "words". However, it's common to see "tokens" still being called "words" anyways.

With tokenization, we redefine our `gpt` function to accept a list of integers and output an integer:

```python
def gpt(inputs: list[int], params) -> int:
    next_word_id = # beep boop, do neural net stuff but now with actual numbers!
    return next_word_id

inputs = tokenizer.encode("not all heroes wear") # [1662, 477, 10281, 5806]
next_word_id = gpt(inputs, params) # 1451
next_word = tokenizer.idx_to_word[next_word_id] # "Ġcap"
```

### Probabilistic
Neural networks model things probabilisticly. As such, `gpt` should output a probability distribution over our vocabulary instead of just outright predicting the next word:

```python
def gpt(inputs: list[int], params) -> list[float]:
    # next_word_prob_dist is a probability distribution, i.e a list of floats where:
    # 1) each number in the list is between 0 and 1
    # 2) the list sums to 1
    # 3) the list of size len(vocab)
    # 3) next_word_prob_dist[i] represents the probability that the next word is word at vocab[i]
    next_word_prob_dist = # beep boop, compute a probability distribution
    return next_word_prob_dist

# assuming our vocab is ["hats", "capes", "heroes", "not", "all", "wear"]
# then our input = [10, 12, 41, 20] = ["not" "all" "heroes" "wear"]
# and our output = [0.20, 0.79, 0.0, 0.0, 0.0, 0.0]
# suggests that our model thinks the next word is:
# - "hats" with a probability of 20%
# - "capes" with a probability of 0.79%
# - "heroes" with a probability of 0%
# - "not" with a probability of 0%
# - "all" with a probability of 1%
# - "wear" with a probability of 0%
gpt([10, 12, 41, 20], params) # [0.20, 0.79, 0.0, 0.0, 0.01, 0.0]
```

To get the next word prediction, we can take the word with the highest probability:

```python
next_token_prob_dist = gpt(inputs, params) # input text = "not all heroes wear"
next_token_id = jnp.argmax(next_token_prob_dist) # "capes"
```

Taking the word with the highest probability as our prediction is often referred to as greedy decoding or greedy sampling.

### Sampling
We can introduce some stochasticity (randomness) to our generations by sampling from the probability distribution instead of being greedy:
```python
next_token_prob_dist = gpt(inputs, params) # input text = "not all heroes wear"
np.random.categorical(next_token_prob_dist) # capes
np.random.categorical(next_token_prob_dist) # hats
np.random.categorical(next_token_prob_dist) # capes
np.random.categorical(next_token_prob_dist) # capes
np.random.categorical(next_token_prob_dist) # pants
```
Not only does it allow us to generate completely different sentences during generation, but it also greatly increases the quailty of the outputs compared to greedy decoding. This is because greedy decoding is prone to repetition and is too safe.

In practice, we also modify the output probability distributions using techniques like [top-p, top-k](https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p), and [temperature](https://docs.cohere.ai/docs/temperature) before sampling them to improve performance. Another nice property of these techniques is they introduce hyperparameters that we can play around with to get different generation behaviours (for example, increasing temperature makes our model take more risks and hence it may become more "creative").

### Multiple Outputs
Becuase of the way the underlying neural network (the transformer) is structured, GPTs don't just output a single probability distribution. Instead, they output `len(inputs)` probability distributions, one for each token in the input:
```python
def gpt(inputs: list[int], params) -> list[float]:
    # next_word_prob_dists is matrix with len(inputs) rows and len(vocab) columns
    # next_word_prob_dists[i] corresponds to probability distribution for the word at position i + 1
    next_word_prob_dists = # beep boop, compute a probability distribution per input
    return next_word_prob_dists

# assuming our vocab is ["hats", "capes", "heroes", "not", "all", "wear"]
# then our input = [10, 12, 41, 20] = ["not" "all" "heroes" "wear"]
output = gpt([10, 12, 41, 20], params) 
output[0] # [0.1, 0.0, 0.0, 0.0, 0.9, 0.0] -> given just the word "not", the model predicts "all"
output[1] # [0.2, 0.0, 0.7, 0.0, 0.1, 0.0] -> given "not all", the model predicts "heroes"
# etc ...
````
To get our the next word for the entire sequence, we just take the last probability distribution in the output:
```python
output[-1] # [0.20, 0.79, 0.0, 0.0, 0.01, 0.0] -> given the whole sequence, the model predicts "capes"
```
This is essentially equivalent to using for loop with out old single output `gpt` function:
```python
next_token_probs_dists = []
for i in range(len(inputs)):
    next_token_prob_dist = gpt(inputs[:i+1])
    next_token_probs_dists.append(next_token_prob_dist)
```
But because of the underlying transformer neural network architecture, we don't need to use a for loop, everything happens in parallel.

### Training
We train a GPT like any other neural network (i.e. gradient descent). The important part is our loss function:

```python
def lm_loss(inputs: list[int], params) -> float:
    # labels are just the input shifted 1 to the left
    #
    # inputs = [not,     all,   heros,   wear,   capes]
    #      x = [not,     all,   heroes,  wear]
    #      y = [all,  heroes,     wear,  capes]
    # 
    # of course, inputs[-1] doesn't have a next word, so we exclude it from x
    # as such, for N inputs, we have N - 1 langauge modeling examples
    x, y = inputs[:-1], inputs[1:]
    
    # forward pass
    # all the predicted next word probability distributions are processed
    # in parallel by the transformer architecture
    prob_dists = gpt(x, params)
    
    # cross entropy loss
    # we take the average over all N-1 examples
    loss = jnp.mean(-jnp.log(prob_dists[y]))

    return loss
```

One importat thing to notice here is that we don't need labelled data. Instead, we are able to produce the labels using the inputs `x, y = inputs[:-1], inputs[1:]`. This is referred to as [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) learning.

This means we can pretty much throw as much text data as we can get our hands at the model (no need for human annotaters). For example, GPT-3 was trained on 300 billion tokens of internet data and books.

![gpt-data](https://miro.medium.com/max/1400/1*Sc3Gi73hepgrOLnx8bXFBA.png)

Being able to scale data to the magnitude of billions of examples is what makes GPT-3 and other LLMs really good.

Of course, you need a sufficiently large model to be able to learn from all this data, which is why GPT-3 is 175 billion parameters and probably cost between [$1m-10m in compute cost to train](https://twitter.com/eturner303/status/1266264358771757057) (although, [175B is probably too big for the amount of data GPT-3 was shown](https://arxiv.org/pdf/2203.15556.pdf)). 

In any case, that's it for this section. We've covered GPT at a high level, and you should now be able to understand the sentence "GPT-3 is just a large transformer based auto regressive language model that was trained on the internet via self-supervised learning".

Now let's get to the fun stuff, the actual GPT network architecture.

## GPT Architecture
### Basic Layers
#### Gelu
[Gaussian Error Linear Units](https://arxiv.org/pdf/1606.08415.pdf) is an alternative to the ReLU activation function, and is approximated by the following function:

```python
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))
```

![gelu](https://miro.medium.com/max/491/1*kwHcbpKUNLda8tvCiwudqQ.png)

There's a [great paper from Noam Shazeer](https://arxiv.org/pdf/2002.05202.pdf) that surveys the performance of various activation functions on the GPT architecture. As far as I can remember, [BERT](https://arxiv.org/pdf/1810.04805.pdf) was the first popular model to use GeLU, and it stuck around for tranformer models.

#### Softmax
Good ole softmax:
```python
def softmax(x, axis=-1):
    exp_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)
```
We use the [`max(x)` trick for numerical stability](https://jaykmody.com/blog/stable-softmax/).

#### Layer Normalization
[Layer normalization](https://arxiv.org/pdf/1607.06450.pdf) normalizes the activation values to have a mean of 0 and a variance of 1. This makes sure the inputs for each layer are within a consitent range, which is suppose to speed up and stabalize the training process:
```python
class LayerNormParams(NamedTuple):
    gamma: Float[Array, "d_model"]
    beta: Float[Array, "d_model"]

def layer_norm(
    x: Float[Array, "seq_len d_model"], params: LayerNormParams, eps: float = 1e-5
) -> Float[Array, "seq_len d_model"]:
    mean = jnp.mean(x, axis=-1)[..., None]
    variance = jnp.var(x, axis=-1)[..., None]
    out = (x - mean) / jnp.sqrt(variance + eps)
    return params.gamma * out + params.beta
```

Like [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf), the normalized output is scaled and offset with two learnable vectors gamma and beta. Layer norm is used instead of batch norm for [various reasons](https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm). The differences between various normalization techniques is outlined [in this excellent blog post](https://tungmphung.com/deep-learning-normalization-methods/).

### Forward Pass
Let's finally write the actual code for our `gpt` function, without any `# beep boop` magic:
```python
class GPTParams(NamedTuple):
    W_e: Float[Array, "vocab_size d_model"]
    W_p: Float[Array, "max_seq_len d_model"]
    blocks: list[TransformerBlockParams]
    ln_f: LayerNormParams

def gpt(
    input_ids: Int[Array, "seq_len"], params: GPTParams, h: int
) -> Float[Array, "seq_len vocab_size"]:
    # token + position embeddings
    # [seq_len] -> [seq_len, d_model]
    seq_len = input_ids.shape[0]
    x = params.W_e[input_ids] + params.W_p[jnp.arange(seq_len)]

    # feed forward through transformer blocks
    for block in params.blocks:
        # [seq_len, d_model] -> [seq_len, d_model]
        x = transformer_block(x, block, h)

    # projection to vocab
    x = layer_norm(x, params.ln_f) # [seq_len, d_model] -> [seq_len, d_model]
    return x @ params.W_e.T # [seq_len, d_model] -> [seq_len, vocab_size] 
```

There are three steps happening here:

1) Embedding the input ids to vectors
2) Feed forward through the multiple transformer block layers
3) Projection to vocab which gives us our output probability distribution (pre-softmax)

This formulation is pretty much directly from the [original GPT paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (see equation 2 under section 3.1).

To better understand what's going on, let's break down each of these 3 steps.

#### Embeddings
**Word Embeddings**
Our token ids are not very good representations for our words. Firstly, a single number is not a lot of dimensionatliy for our model to work with. Secondly, the relative magnitudes of the token ids is falsely communicating information. For example, if `Apple = 5` and `Table = 10` in our vocab, then we are implying that `2 * Table = Apple`??? That doesn't make very much sense.

One way to overcome this is to encode our inputs as one-hot vectors:
```python
# given vocab = ["the", "apple", "banana", "pear", "table"]
one_hot_embedding_for_apple = [0, 1, 0, 0, 0]
one_hot_embedding_for_pear  = [0, 0, 0, 1, 0]
# etc ...
```
These representations are now categorical, removing the magnitude information all together.

However these vectors are now too large (for GPT-2, the BPE encoder vocabulary is 50k words so a sentence of just 100 tokens would require a hundred 50000 sized vectors!). Even worse, one-hot vectors are extremely information sparse. Most of the entries are 0, and our network might struggle to learn anything at all.

Instead, we'll using the idea of [word vectors](https://jaykmody.com/blog/attention-intuition/#word-vectors-and-similarity) and use a learned embedding.
```python
word_embeddings = params.W_e[input_ids]
```
Here our matrix  `W_e` is a `vocab_size by d_model` matrix and acts as a lookup table. That is, each row corresponds to the learned word vector for the ith word in our vocabulary. So if we pass a list of $N$ `input_ids` as indices, we are just retrieving the $N$ word vectors associated with those ids.

`d_model` is a hyperparameter we choose that determines the dimensionality of our word vectors. Larger values enable our model to encode more information about a word, and is our primary way of increasing the "width" of our network.

Like any other parameter in our network, `W_e` is learned. That is, it is randomly initialized at the start of training and then updated via gradient descent.

**Positional Embeddings**
One quirk of the transformer architecture is that it doesn't actually care about position. If we randomly shuffled our input and then accordinly unshuffled our output, the output would look the exact same (meaning position plays no role). Of course, the order of words in language is super important so we need some way of encoding order into our inputs. For this, we can just use another learned embedding matrix:

```python
seq_len = input_ids.shape[0]
pos_embeddings = params.W_p[jnp.arange(seq_len)]
```
`W_p` is a  `max_seq_len by d_model` matrix. Each row encodes positional information for position `i` in a sequence. Like `W_e`, this matrix is learned during gradient descent.

You'll notice that this restricts our model to a maximum sequence length. The original transfomer paper used a [calculated positional embedding](https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding) which they found performed just as well as learned positional embeddings, but have the distinct advantage that you can input any arbitrarily long sequence (you are not restricted by a maximum sequence length). However, in practice, your model is only going to be good sequence lengths that the model was trained on. You can't just train a GPT on 1024 long tokens with calculated position embeddings and then expect it to perform well at 16k tokens long.

**Combined**
Adding our word embeddings and positional embeddings together, we now have a way to convert our sequence of input ids into rich vector representations that can encode both the meaning and position of words.

```python
# token + position embeddings
# [seq_len] -> [seq_len, d_model]
seq_len = input_ids.shape[0]
x = params.W_e[input_ids] + params.W_p[jnp.arange(seq_len)]

# x[i] represents the word embedding for the ith word + the positional
# embedding for the ith position
```

#### Transformer
This is where all the magic happens and the "deep" in deep learning comes in. We pass our embedding through a stack of `n_layers` transformer blocks.
```python
# feed forward through transformer blocks
for block in params.blocks:
    # [seq_len, d_model] -> [seq_len, d_model]
    x = transformer_block(x, block, h)
```
Stacking more layers is what allows us to control how "deep" our networks is. GPT-3 for example, has a [whopping 96 layers](https://preview.redd.it/n9fgba8b0qr01.png?auto=webp&s=e86d2d3447c777d3222016e81a0adfaec1a95592).

#### Projection to Vocab
The output from the last transformer block should contain really rich representations after going through so many layers. So in our final step, all we need to do is project our vectors to a probability distribution over our vocab:

```python
# projection to vocab
x = layer_norm(x, params.ln_f) # [seq_len, d_model] -> [seq_len, d_model]
return x @ params.W_e.T # [seq_len, d_model] -> [seq_len, vocab_size] 
```

Couple things to note here:
1. We first pass `x` through a final layer normalization layer before doing the projection to vocab. This is specific to the GPT-2 architecture and is not something was done in the original Transformer paper.
2. Our projection to vocab `x @ params.W_e.T` maps our `d_model` dimension to `vocab_size`, giving us our desired output shape `[seq_len, vocab_size]`. The symbol `@` means matrix multiplication.
3. We are reusing the embedding matrix `W_e` (but transposed) to do our projection. You could instead use a seperate weight matrix `W_y` initialized to `[d_model, vocab_size]`. Sharing the embedding matrix was also something the original transformer did. I guess the advantage of this is 1) you save some parameters and 2) since you are the matrix to both be able to map to words and from words, it should learn a richer representation and improve learning. However, at the scale of GPT-3, this doesn't actually save too many parameters (the embedding matrix basically becomes negligible in size compared to the 96 monstrous transformer block layers).
4. We don't take the softmax at the end, so our outputs will be [logits](https://developers.google.com/machine-learning/glossary/#logits) instead of probabilities between 0 and 1. Why? This is for numerically stability reasons. For example, to compute cross entropy loss, taking [`log(softmax(logits))` is numerically instable compared to `log_sofmtax(logits)`](https://jaykmody.com/blog/stable-softmax/). As This is why `torch.nn.CrossEntropy` accepts logits and not probabilities. Lots of other functions also accept logits instead of probabilities, for example, `jax.random.categorical`. In any case, we can always convert logits to probabilties by taking the softmax, but we can't convert probabilities back to logits, so it is more flexible to output logits.

### Transformer Block
Let's dig a litter deeper into what `transformer_block` is doing:
```python
class TransformerBlockParams(NamedTuple):
    mha_params: MultiHeadAttentionParams
    ffn_params: PositionWiseFFNParams
    ln_1_params: LayerNormParams
    ln_2_params: LayerNormParams


def transformer_block(
    x: Float[Array, "seq_len d_model"], params: TransformerBlockParams, h: int
) -> Float[Array, "seq_len d_model"]:
    x = x + casual_multi_head_self_attention(layer_norm(x, params.ln_1_params), params.mha_params, h)
    x = x + position_wise_ffn(layer_norm(x, params.ln_2_params), params.ffn_params)
    return x
```
This looks a bit crowded with all the passing around of parameteres. Let's refactor a little bit to make it easier to read:
```python
def transformer_block(
    x: Float[Array, "seq_len d_model"], params: TransformerBlockParams, h: int
) -> Float[Array, "seq_len d_model"]:
    mha = lambda x: casual_multi_head_self_attention(x, params.mha_params, h)
    ffn = lambda x: position_wise_ffn(x, params.ffn_params)
    ln_1 = lambda x: layer_norm(x, params.ln_1_params)
    ln_2 = lambda x: layer_norm(x, params.ln_2_params)

    x = x + mha(ln_1(x))
    x = x + ffn(ln_2(x))
```
Here, we can clearly tell whats happening from the last two lines:

1. First, we pass our input through multi-head attention
2. Then, we pass our input through a feed forward neural network
3. Both "sublayers" use a residual connection (i.e. we add the input to the output of the sublayer)
4. Both also use layer normalization on the input

The purpose of each of these:

1. The multi-head attention is what facilitates the communication between the inputs. Nowhere else in the network does the model allow the inputs to communicate with each other. The embeddings, position-wise feed forward network, layer norms, and projection to vocab all operate on our inputs position-wise. Inter-input information sharing is tasked solely to this layer.
2. The position-wise feed forward network is just a regular multi-layer perceptron. This just adds a bunch of learnable parameters for our model to work with to faciliatate learning.
3. Residual connections (popularized by [ResNet](https://arxiv.org/pdf/1512.03385.pdf)) have three main advantages:
    1. Makes it easier to optimize neural networks that are deep (i.e. lots of layers). The idea here is that we are providing "shortcuts' for the gradients to flow back through the network.
    2. Without residual connections, deeper models see a degredation in performance when adding more layers (possibly because it's hard for the gradients to flow all the way back through a deep network without losing information). Residual connections seem to give a bit of an accuracy boost for deeper networks.
    3. Can help with the [vanishing/exploding gradients problem](https://programmathically.com/understanding-the-exploding-and-vanishing-gradients-problem/).
4. We've already discussed the advantages of layer normalization, so I won't repeat it here. What is worth noting however is that in the original transformer paper, they do layer norm on the output `layer_norm(x + sublayer(x))` while we do layer norm on the inputs `x + sublayer(layer_norm(x))` to match GPT-2. This is sometimes referred to as pre-norm and has been shown to be [important in improving the perfomance of the transformer](https://arxiv.org/pdf/2002.04745.pdf).

### Position-wise Feed Forward Network
This is just a simple multi-layer perceptron with 2 layers:

```python
class PositionWiseFFNParams(NamedTuple):
    W1: Float[Array, "d_model d_ff"]
    b1: Float[Array, "d_ff"]
    W2: Float[Array, "d_ff d_model"]
    b2: Float[Array, "d_model"]


def position_wise_ffn(
    x: Float[Array, "seq_len d_model"], params: PositionWiseFFNParams
) -> Float[Array, "seq_len d_model"]:
    return gelu(x @ params.W1 + params.b1) @ params.W2 + params.b2
```

Nothing super fancy here, we just project from `d_model` up to a higher dimension `d_ff` and then back down to `d_model`. 

`d_ff` is a hyperparameter of our choosing, and should be set to a value higher than `d_model`. GPT-2 always uses  `d_ff = 4 * d_model`.

It's also worth noting, we give the multi-head attention a lot of attention (pun intended) for being the backbone of the transformer, but at the scale of GPT-3, [80% of the model parameters are in contained in the feed forward layer](https://twitter.com/stephenroller/status/1579993017234382849). Just something to think about.

### Multi-Head Casual Self Attention
Multi-head attention is the most difficult but most important part of the transformer to understand. Attention is the only place in the Transformer where our inputs are allowed to "share" information.

Let's break down each word into it's own section and work our way up from there:

1) Attention
2) Self
3) Casual
4) Multi-Head

#### Attention
There are some already great explanations of attention like [Lilian Weng's Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) and [Jay Alammar's The Illustrated Transformer](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/).

I [also wrote a blog post](https://jaykmody.com/blog/attention-intuition/) on the topic where I derive the scaled dot product equation proposed in the [original transformer paper](https://arxiv.org/pdf/1706.03762.pdf) from the ground up:
$$\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

As such, I'm going to skip explanationing attention here, you can go read my blog post or some other explanation of attention. As a baseline, we'll use the attention implementation from my blog post:

```python
def attention(Q, K, V):
    # Q is a matrix of shape (n_q, d_k)
    # K is a matrix of shape (n_k, d_k)
    # v is a matrix of shape (n_k, d_v)
    # output is a matrix of shape (n_q, d_v)
    d_k = K.shape[-1]
    return softmax(Q @ K.T / jnp.sqrt(d_k)) @ V
```

#### Self
When Q, K, anv V all come from the same source, we are performing self-attention (i.e. letting some sequence attend to itself):

```python
def self_attention(
    x: Float[Array, "seq_len d_model"]
) -> Float[Array, "seq_len d_model"]:
    return attention(q=x, k=x, v=x)
```

Letting our inputs attent to themselves is what allows them to share information about each other and enables our transformer to model sequences. 

We can enhance self attention by introducting some learning via projections for q, k, v and the attention output:

```python
def self_attention(
    x: Float[Array, "seq_len d_model"]
) -> Float[Array, "seq_len d_model"]:
    # qkv prejoctions
    q = x @ params.W_q # seq_len, d_model @ d_model, d_model = seq_len, d_model
    k = x @ params.W_k # seq_len, d_model @ d_model, d_model = seq_len, d_model
    v = x @ params.W_v # seq_len, d_model @ d_model, d_model = seq_len, d_model

    # perform self attention
    x = attention(q=q, k=k, v=v) # seq_len, d_model -> seq_len, d_model

    # out projection
    x = x @ params.W_out # seq_len, d_model @ d_model, d_model = seq_len, d_model

    return x
```

Not only does this add more parameters for our model to work with, but this is particularly helpful since q, k, and v can have different values (we are still performing self attention, qkv are still derived from the same source, but now our model can choose how to modify q, k, and v to best help it learn relationships between inputs).

We can reduce the number of matrix multiplcations from 4 to just 2 if we combine `W_q`, `W_k` and `W_v` into a single matrix `W_qkv`, perform the projection, and then split the result:

```python
def self_attention(
    x: Float[Array, "seq_len d_model"]
) -> Float[Array, "seq_len d_model"]:
    # qkv prejection
    # seq_len, d_model @ d_model, 3*d_model = seq_len, 3*d_model
    x = x @ params.W_qkv 

    # split into qkv
    # seq_len, 3*d_model -> 3 matrices of seq_len, d_model
    q, k, v = jnp.split(x, 3, axis=-1)

    # perform self attention
    x = attention(q, k, v) # seq_len, d_model -> seq_len, d_model

    # out projection
    x = x @ params.W_out # seq_len, d_model @ d_model, d_model = seq_len, d_model

    return x
```

This is a bit more efficient as modern accelerators (GPUs) can take better advantage of one large matrix multiplcation rather than 3 seperate small ones.

Finally, to match the implementation of GPT-2, we add bias vectors for both the qkv and out projections:

```python
class MultiHeadAttentionParams(NamedTuple):
    W_qkv: Float[Array, "d_model 3*d_model"]
    b_qkv: Float[Array, "3*d_model"]
    W_out: Float[Array, "d_model d_model"]
    b_out: Float[Array, "d_model"] 

def self_attention(
    x: Float[Array, "seq_len d_model"], params: MultiHeadAttentionParams
) -> Float[Array, "seq_len d_model"]:
    # qkv prejection
    # seq_len, d_model @ d_model, 3*d_model = seq_len, 3*d_model
    x = x @ params.W_qkv + params.b_qkv

    # split into qkv
    # seq_len, 3*d_model -> 3 matrices of seq_len, d_model
    q, k, v = jnp.split(x, 3, axis=-1)

    # perform self attention
    x = attention(q, k, v) # seq_len, d_model -> seq_len, d_model

    # out projection
    x = x @ params.W_out # seq_len, d_model @ d_model, d_model = seq_len, d_model

    return x
```

#### Casual
There is a bit of an issue with our current self-attention setup, our inputs can see into the future! For example, if our input is `["not", "all", "heroes", "wear", "capes"]`, during self attention we are allowing "wear" to see "capes". This means our output probability distribution for "wear" will biased since it already knows the answer is "capes" (it's cheating, it is looking at the answer!). This is no good since our model will just learn that the correct answer for input $i$ is found at position $i+1$.

To prevent this, we need to somehow modify our attention matrix `softmax(Q @ K.T / jnp.sqrt(d_k))` to "hide" or _"mask"_  our inputs from being able to see into the future. For example, let's pretend our attention matrix looks like this:
```
       not    all    heroes wear   capes
   not 0.116  0.159  0.055  0.226  0.443
   all 0.180  0.397  0.142  0.106  0.175
heroes 0.156  0.453  0.028  0.129  0.234
  wear 0.499  0.055  0.133  0.017  0.295
 capes 0.089  0.290  0.240  0.228  0.153
```
Each row corresponds to the queries, the columns are the keys that the queries are attending to. In this case, looking at the "wear" row, you can see that it is attending to "capes" in the last column with a weight of 0.295. To prevent this, we want to set that entry to 0.0:
```text
      not    all    heroes wear   capes
   not 0.116  0.159  0.055  0.226  0.443
   all 0.180  0.397  0.142  0.106  0.175
heroes 0.156  0.453  0.028  0.129  0.234
  wear 0.499  0.055  0.133  0.017  0.
 capes 0.089  0.290  0.240  0.228  0.153
```
In general, for any entry $i, j$ where $j > i$, we want the attention matrix to be 0 at that entry:
```
       not    all    heroes wear   capes
   not 0.116  0.     0.     0.     0.
   all 0.180  0.397  0.     0.     0.
heroes 0.156  0.453  0.028  0.     0.
  wear 0.499  0.055  0.133  0.017  0.
 capes 0.089  0.290  0.240  0.228  0.153
```
We call this _masking_. However, the issue with our above approach is our rows no longer sum to 1 (since we are setting them to 0 after the softmax has already normalized things). To make sure our rows still sum to 1, we need to modify our attention matrix before the softmax.

This can be achieved by setting entries that are to be masked with $-\infty$ prior to the softmax[^softmax]:
```python
def attention(Q, K, V, mask):
    # attention matrix pre-softmax
    d_k = K.shape[-1]
    attn = Q @ K.T / jnp.sqrt(d_k)  

    # mask positions
    #
    # here, mask is a (n_q, n_k) matrix of 1s and 0s, 1s corresponding
    # to positions we want to keep, 0s corresponding to positions we want to mask
    # 
    # for numerical stability, we use a reall large negative number rather 
    # than -jnp.inf
    attn = mask * attn - 1e10 * (1 - mask)

    # softmax and value matmul
    attn = softmax(attn)
    return attn @ V
```
where the variable `mask` is (for an input with `seq_len = 5`):
```
1 0 0 0 0 
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```
which is just a lower triangular matrix (positions $j > i$ are set to 0), and can be compute with `jnp.tri(seq_len)`. This type of masking has many names: Casual masking, Subsequent masking, Illegal connections masking, Look ahead masking, Peek ahead masking. We'll just refere to it as casual masking, since this is what enables our model to be casual (without cheating).

Putting it all together, we get:
```python
def attention(
    Q: Float[Array, "n_q d_k"],
    K: Float[Array, "n_k d_k"],
    V: Float[Array, "n_k d_v"],
    mask: Bool[Array, "n_q n_k"],
) -> Float[Array, "n_q d_v"]:
    d_k = K.shape[-1]
    attn = Q @ K.T / jnp.sqrt(d_k)
    attn = mask * attn - 1e10 * (1 - mask)
    attn = softmax(attn)
    return attn @ V

def casual_self_attention(
    x: Float[Array, "seq_len d_model"], params: MultiHeadAttentionParams
) -> Float[Array, "seq_len d_model"]:
    # qkv prejection
    # seq_len, d_model @ d_model, 3*d_model = seq_len, 3*d_model
    x = x @ params.W_qkv + params.b_qkv

    # split into qkv
    # seq_len, 3*d_model -> 3 matrices of seq_len, d_model
    q, k, v = jnp.split(x, 3, axis=-1)

    # casual mask
    seq_len = q.shape[1]
    casual_mask = jnp.tri(seq_len)

    # perform self attention
    x = attention(q, k, v, casual_mask) # seq_len, d_model -> seq_len, d_model

    # out projection
    x = x @ params.W_out # seq_len, d_model @ d_model, d_model = seq_len, d_model

    return x
```

#### Multi-Head
We can further improve the capabilities our attention layer by splitting our queries, keys, and values into `h` different _"heads"_ and doing `h` seperate attention calculations instead of just the one:
```python
def multi_head_casual_self_attention(
    x: Float[Array, "seq_len d_model"], params: MultiHeadAttentionParams, h: int
) -> Float[Array, "seq_len d_model"]:
    # qkv projection
    # [seq_len, d_model] -> [seq_len, 3*d_model]
    x = x @ params.W_qkv + params.b_qkv

    # split into qkv
    # [seq_len, 3*d_model] -> 3 of [seq_len, d_model]
    q, k, v = jnp.split(x, 3, axis=-1)

    # split into heads
    # 3 of [seq_len, d_model] -> 3 of [h, seq_len, d_model/h]
    q, k, v = map(lambda x: jnp.array(jnp.split(x, h, axis=-1)), [q, k, v])

    # casual mask
    seq_len = q.shape[1]
    casual_mask = jnp.tri(seq_len, dtype=bool)  # [seq_len, seq_len]

    # perform attention for each head
    heads = []
    for i in range(h):
        heads = attention(q[i], k[i], v[i], casual_mask)  # [h, seq_len, d_model/h]

    # merge heads
    # [h, seq_len, d_model/h] -> [seq_len, d_model]
    x = jnp.hstack(heads)

    # output projection
    # [seq_len, d_model] -> [seq_len, d_model]
    return x @ params.W_out + params.b_out
```
There are 3 added steps here:

1) Split `q, k, v` into `h` heads
```python
# split into heads
# 3 of [seq_len, d_model] -> 3 of [h, seq_len, d_model/h]
q, k, v = map(lambda x: jnp.array(jnp.split(x, h, axis=-1)), [q, k, v])
```
2) Perform `h` seperate attention calculations for each head
```python
# perform attention for each head
heads = []
for i in range(h):
    heads = attention(q[i], k[i], v[i], casual_mask)  # [h, seq_len, d_model/h]
```
3) Merge the output of each head
```python
# merge heads
# [h, seq_len, d_model/h] -> [seq_len, d_model]
x = jnp.hstack(heads)
```

One not so nice thing is the new and introduced for loop. This is inefficient, each head is independent so we can calculate them in parallel. To do this, we can take advantage of [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) instead of using a for loop:
```python
# perform attention for each head
attn_fn = jax.vmap(attention, in_axes=(0, 0, 0, None))
heads = attn_fn(q, k, v, casual_mask)  # [h, seq_len, d_model/h]
```

## Testing our Implementation
Putting all our model code into a single file, we get [model.py](https://github.com/jaymody/gpt-jax/blob/main/model.py#).

To do a test forward pass, we can initialize our model with random parameters, pass some pretend inputs, and verify the output shape is correct as seen in [`test.py`](https://github.com/jaymody/gpt-jax/blob/main/tests.py#L20).

To generate entire sentences, we load [download the official GPT-2 model checkpoint from OpenAI](https://github.com/jaymody/gpt-jax/blob/19c5d67e0501c278f4185ac283aff454ac681ff2/main.py#L24-L58), [map them to work with our implementation](https://github.com/jaymody/gpt-jax/blob/19c5d67e0501c278f4185ac283aff454ac681ff2/main.py#L61-L110), and then [use](https://github.com/jaymody/gpt-jax/blob/19c5d67e0501c278f4185ac283aff454ac681ff2/main.py#L113-L161) the [`generate`](https://github.com/jaymody/gpt-jax/blob/19c5d67e0501c278f4185ac283aff454ac681ff2/sampling.py#L11) function to auto-regressively sample from the model.

For example, running:
```python
poetry python main.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --model_type "124M" \
    --n_tokens_to_generate 40 \
    --greedy
```
Gives the output:
```
the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

Which is [equivalent to what you would obtain running `openai/gpt-2`](https://github.com/jaymody/gpt-jax#correctness).

## Discussion

TODO

* Why the transformer? Why not some other architecture? What's so special about it?
* How did we get here? What are the papers and ideas that got us to GPT-3?
* What kind of things can LLMs do?
* Are there any tricks to training a GPT? How are we able to train it at scale?
* What is prompt engineering?
* What is AI Alignment?
* What is finetuning?
* Further reading
