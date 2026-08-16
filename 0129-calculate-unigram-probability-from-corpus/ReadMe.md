# Calculate Unigram Probability from Corpus (Easy, NLP)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding Unigram Probability](#learn-understanding-unigram-probability)
- [Solution](#solution)
  - [Custom Implementation](#custom-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Calculate Unigram Probability from Corpus](https://www.deep-ml.com/problems/129)

Implement a function that calculates the **unigram probability** of a given word in a corpus of sentences.

The corpus is represented as a string containing space-separated tokens. The special start-of-sentence `<s>` and end-of-sentence `</s>` tokens must also be included when calculating the total number of tokens.

The probability should be rounded to **4 decimal places**.

---

## Example

### Input

```python
corpus = "<s> Jack I like </s> <s> Jack I do like </s>"
word = "Jack"
```

### Output

```text
0.1818
```

### Reasoning

The corpus contains **11 total tokens**, including the `<s>` and `</s>` tokens.

The word `Jack` appears **2 times**.

Therefore:

$$
P(\text{Jack})=\frac{\text{Count}(\text{Jack})}{\text{Total Tokens}}
$$

$$
P(\text{Jack})=\frac{2}{11}=0.1818
$$

Hence, the unigram probability is `0.1818`.

---

## Learn: Understanding Unigram Probability

### What is it?

A **unigram** is a single token considered independently from the surrounding tokens.

A **unigram language model** is the simplest form of a statistical language model. It estimates the probability of each token based only on how frequently that token occurs in the corpus.

The model makes an independence assumption: the probability of a token does not depend on the previous or following tokens.

For a word $w$, its unigram probability is determined by its frequency relative to the total number of tokens.

---

### Mathematical Definition

The probability of a word $w$ is:

$$
P(w)=\frac{\text{Count}(w)}{\sum_{w'\in V}\text{Count}(w')}
$$

where:

- $\text{Count}(w)$ is the number of occurrences of word $w$.
- $V$ is the vocabulary containing all tokens.
- $\sum_{w'\in V}\text{Count}(w')$ is the total number of tokens in the corpus.

Therefore, the calculation is simply:

$$
P(w)=\frac{\text{Number of occurrences of }w}{\text{Total number of tokens}}
$$

---

### Sentence Boundary Tokens

NLP corpora often include special tokens to represent sentence boundaries:

- `<s>` represents the **start of a sentence**.
- `</s>` represents the **end of a sentence**.

For this problem, both tokens are treated as normal tokens when calculating the total token count.

For example:

```text
<s> I am Jack </s>
```

contains five tokens:

```text
<s>
I
am
Jack
</s>
```

Therefore, sentence boundary tokens contribute to the denominator of the unigram probability.

---

### Tokenization

Before calculating the probability, the corpus needs to be divided into individual tokens.

For a corpus such as:

```text
<s> Jack I like </s> <s> Jack I do like </s>
```

splitting on whitespace produces:

```text
<s>
Jack
I
like
</s>
<s>
Jack
I
do
like
</s>
```

There are therefore 11 tokens.

The frequency of `Jack` is:

$$
\text{Count}(\text{Jack})=2
$$

and the total number of tokens is:

$$
N=11
$$

so:

$$
P(\text{Jack})=\frac{2}{11}=0.1818
$$

---

### Unigram Model Assumption

The defining assumption of a unigram model is that tokens are treated independently.

For a sequence of tokens:

$$
w_1,w_2,\ldots,w_n
$$

the probability of the sequence is approximated as:

$$
P(w_1,w_2,\ldots,w_n)=\prod_{i=1}^{n}P(w_i)
$$

This ignores relationships between neighboring words.

For example, the unigram model does not explicitly consider that `machine` is frequently followed by `learning`.

It only considers the individual frequencies of `machine` and `learning`.

---

### Example with a Small Corpus

Consider:

```text
<s> I am Jack </s>
<s> Jack I am </s>
```

The tokens are:

```text
<s> I am Jack </s> <s> Jack I am </s>
```

There are 10 total tokens.

`Jack` occurs twice:

$$
P(\text{Jack})=\frac{2}{10}=0.2
$$

Similarly, `I` also occurs twice:

$$
P(\text{I})=\frac{2}{10}=0.2
$$

while `am` occurs twice:

$$
P(\text{am})=\frac{2}{10}=0.2
$$

The probability depends entirely on token frequency.

---

### Probability Distribution

If the corpus contains all tokens in the vocabulary, their unigram probabilities form a probability distribution.

Therefore:

$$
\sum_{w\in V}P(w)=1
$$

because:

$$
\sum_{w\in V}P(w)=\sum_{w\in V}\frac{\text{Count}(w)}{N}
$$

which becomes:

$$
\frac{\sum_{w\in V}\text{Count}(w)}{N}=1
$$

where $N$ is the total number of tokens.

---

### Zero Probability

If a word does not occur in the corpus, its observed unigram probability is:

$$
P(w)=0
$$

This creates an important problem for language modeling.

When probabilities are multiplied to estimate the probability of a sentence, a single unseen word can make the entire sentence probability zero.

This is known as the **zero-frequency problem**.

The standard solution is to use **smoothing techniques**, such as Laplace smoothing, which assign a small non-zero probability to unseen tokens.

This problem does not require smoothing.

---

### Characteristics / Key Points

- A unigram consists of one token.
- A unigram model treats tokens independently.
- Probability depends only on token frequency.
- The denominator is the total number of tokens.
- `<s>` and `</s>` are counted as tokens in this problem.
- A word appearing more frequently receives a higher probability.
- An unseen word has probability zero without smoothing.
- Unigram probabilities form a probability distribution over the vocabulary.
- The probabilities of all vocabulary tokens sum to $1$.
- Tokenization determines what is counted as an individual token.

---

### Why is it used? / Applications

Unigram probabilities are useful as a foundation for statistical NLP.

Applications include:

- Language modeling.
- Text classification.
- Spam detection.
- Word frequency analysis.
- Text generation.
- Baseline NLP models.
- Probability estimation.
- Smoothing techniques.
- Comparing token frequencies.

Although modern NLP systems commonly use more sophisticated models, unigram models are important for understanding the foundations of probabilistic language modeling.

N-gram models extend this idea by considering sequences of multiple tokens:

- **Unigram:** one token.
- **Bigram:** two consecutive tokens.
- **Trigram:** three consecutive tokens.

For example, a bigram model considers:

$$
P(w_i\mid w_{i-1})
$$

while a unigram model simply estimates:

$$
P(w_i)
$$

---

> 💡 **Important Note**
>
> `<s>` and `</s>` are not ignored in this problem. They are part of the corpus token sequence and therefore contribute to the total token count. Forgetting them changes the denominator and produces an incorrect probability.

> 💡 **Interview Tip**
>
> For a basic unigram probability problem, think in two steps: **count the target token** and **count all tokens**. Then divide the first count by the second and round the result.

---

## Solution

### Custom Implementation

```python
def unigram_probability(corpus: str, word: str) -> float:
    tokens = corpus.split()
    return round(sum(token == word for token in tokens) / len(tokens), 4)
```

---

## Code Explanation

### Step 1: Tokenize the Corpus

The corpus is split into individual tokens:

```python
tokens = corpus.split()
```

Python's `split()` separates the string using whitespace.

For example:

```text
<s> Jack I like </s>
```

becomes:

```python
["<s>", "Jack", "I", "like", "</s>"]
```

This automatically preserves `<s>` and `</s>` as individual tokens.

---

### Step 2: Count the Target Word

The expression:

```python
sum(token == word for token in tokens)
```

checks every token against the requested word.

For each token:

- `True` means the token matches.
- `False` means it does not.

Python treats `True` as `1` and `False` as `0` when summed.

Therefore, the expression directly gives the number of occurrences of `word`.

For example:

```python
sum(token == "Jack" for token in tokens)
```

returns:

```text
2
```

---

### Step 3: Count Total Tokens

The total number of tokens is:

```python
len(tokens)
```

For the example corpus:

```text
<s> Jack I like </s> <s> Jack I do like </s>
```

the result is:

```text
11
```

---

### Step 4: Calculate the Probability

The probability is calculated as:

```python
word_count / len(tokens)
```

which corresponds directly to:

$$
P(w)=\frac{\text{Count}(w)}{\text{Total Tokens}}
$$

For `Jack`:

$$
P(\text{Jack})=\frac{2}{11}
$$

---

### Step 5: Round the Result

The result is rounded to four decimal places:

```python
round(..., 4)
```

Therefore:

$$
\frac{2}{11}=0.1818
$$

and the function returns:

```text
0.1818
```

---

### Algorithm

1. Split the corpus into tokens.
2. Count how many tokens match the target word.
3. Count the total number of tokens.
4. Divide the target count by the total count.
5. Round the probability to four decimal places.
6. Return the result.

---

## Time & Space Complexity

Let $N$ be the total number of tokens in the corpus.

| Complexity | Value    |
| ---------- | -------- |
| Time       | **O(N)** |
| Space      | **O(N)** |

The corpus must be traversed to tokenize it and count occurrences, resulting in **O(N)** time.

The `split()` operation stores all tokens in a list, requiring **O(N)** auxiliary space.
