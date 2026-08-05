# Implement TF-IDF (Term Frequency-Inverse Document Frequency) (Medium, NLP)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Understanding TF-IDF (Term Frequency-Inverse Document Frequency)](#learn-understanding-tf-idf-term-frequency-inverse-document-frequency)
- [Solution](#solution)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

### [Implement TF-IDF (Term Frequency-Inverse Document Frequency)](https://www.deep-ml.com/problems/60)

Write a Python function that computes the **TF-IDF (Term Frequency-Inverse Document Frequency)** scores for a given query over a corpus of documents.

The function should:

- Accept a corpus where each document is represented as a list of words.
- Accept a query consisting of one or more words.
- Compute the **Term Frequency (TF)** for every query word in every document.
- Compute the **Inverse Document Frequency (IDF)** using smoothing.
- Multiply TF and IDF to obtain the TF-IDF score.
- Return the scores rounded to **five decimal places**.
- Handle empty documents, empty corpus, and query words not present in the corpus.

---

## Example

### Input

```python
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "bird", "flew", "over", "the", "mat"]
]

query = ["cat"]

print(compute_tf_idf(corpus, query))
```

### Output

```python
[[0.21461], [0.25754], [0.0]]
```

### Reasoning

The word **"cat"** appears in **2** of the **3** documents.

Its IDF is computed using the smoothed formula, while the TF is calculated independently for each document.

Multiplying the TF and IDF values gives

- Document 1 → **0.21461**
- Document 2 → **0.25754**
- Document 3 → **0.0**

---

## Learn: Understanding TF-IDF (Term Frequency-Inverse Document Frequency)

### What is it?

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure that indicates how important a word is to a document relative to an entire collection of documents (called a **corpus**).

The idea is simple:

- Words that appear frequently in a document are important.
- Words that appear in almost every document are less informative.
- Rare but frequent words receive the highest importance.

TF-IDF is one of the most widely used feature extraction techniques in **Natural Language Processing (NLP)** and **Information Retrieval**.

---

### Mathematical Definition

TF-IDF is the product of **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

$$ \text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t) $$

where

- $t$ is the term.
- $d$ is a document.
- $N$ is the total number of documents.

---

### Term Frequency (TF)

Term Frequency measures how often a word appears inside a document.

$$ \text{TF}(t,d) = \frac{\text{Count of }t\text{ in document }d}{\text{Total words in document }d} $$

A higher TF means the word is more important within that document.

Example:

Document

```text
the cat sat on the mat
```

Word **cat**

- Frequency = 1
- Total words = 6

Therefore,

$$ \text{TF}(\text{cat},d) = \frac{1}{6} \approx 0.1667 $$

---

### Inverse Document Frequency (IDF)

Some words such as

- the
- is
- and
- of

appear in almost every document.

Although they have high TF, they carry little useful information.

IDF reduces the importance of such common words.

The smoothed IDF formula is

$$ \text{IDF}(t) = \log\left(\frac{N+1}{df(t)+1}\right)+1 $$

where

- $N$ is the total number of documents.
- $df(t)$ is the number of documents containing the term.
- Adding **1** to the numerator and denominator prevents division by zero.
- Adding **1** outside the logarithm keeps the IDF value positive.

---

### TF-IDF Calculation

Once TF and IDF are available,

$$ \text{TF-IDF}(t,d) = \text{TF}(t,d)\times\text{IDF}(t) $$

Words that are frequent within one document but rare across the corpus receive the highest TF-IDF scores.

---

### Example Calculation

Consider the corpus

- Doc1: "The cat sat on the mat"
- Doc2: "The dog chased the cat"
- Doc3: "The bird flew over the mat"

We compute the TF-IDF of **cat** in Doc1.

#### Step 1

Compute TF.

The word appears once among six words.

$$ \text{TF}(\text{cat},Doc_1) = \frac{1}{6} \approx 0.1667 $$

#### Step 2

Compute document frequency.

The word appears in two documents.

$$ df(\text{cat}) = 2 $$

Total documents

$$ N = 3 $$

#### Step 3

Compute IDF.

$$ \text{IDF}(\text{cat}) = \log\left(\frac{3+1}{2+1}\right)+1 = \log\left(\frac{4}{3}\right)+1 \approx 1.2877 $$

#### Step 4

Multiply TF and IDF.

$$ \text{TF-IDF}(\text{cat},Doc_1) = 0.1667 \times 1.2877 \approx 0.2146 $$

---

### Why Smoothing is Important

Without smoothing,

$$ \text{IDF}(t) = \log\left(\frac{N}{df(t)}\right) $$

If

$$ df(t) = 0 $$

division by zero occurs.

Using

$$ \text{IDF}(t) = \log\left(\frac{N+1}{df(t)+1}\right)+1 $$

avoids this issue while keeping the metric numerically stable.

---

### Characteristics / Key Points

- Measures word importance relative to a corpus.
- Combines local importance (TF) with global rarity (IDF).
- Common words receive smaller weights.
- Rare but meaningful words receive larger weights.
- Produces sparse feature vectors.
- Easy to compute and interpret.
- Independent of word order.
- Frequently used before applying machine learning algorithms.

---

### Advantages

- Simple to implement.
- Computationally efficient.
- Works well for traditional NLP pipelines.
- Improves document ranking.
- Highlights informative keywords.
- Reduces the influence of stop words.

---

### Limitations

- Ignores semantic meaning.
- Treats every word independently.
- Does not consider word order.
- Cannot capture context.
- Performs poorly with synonyms.
- Modern transformer models often outperform TF-IDF for semantic tasks.

---

### Applications

TF-IDF is widely used in

- Search Engines
- Information Retrieval
- Text Classification
- Spam Detection
- Document Ranking
- Topic Extraction
- Keyword Extraction
- Recommendation Systems
- Document Similarity
- Text Clustering

It remains one of the most important classical NLP techniques.

---

### Practical Example

Suppose two documents contain

```text
Document A:
deep learning neural networks

Document B:
deep learning machine learning
```

The word **deep** appears in both documents, so its IDF becomes small.

The word **neural** appears only in Document A, giving it a much larger IDF.

As a result, **neural** contributes more to distinguishing Document A from the rest of the corpus.

---

### Common Mistakes

- Forgetting to smooth the IDF calculation.
- Computing TF using the total corpus size instead of document size.
- Ignoring empty documents.
- Not handling query words absent from the corpus.
- Forgetting to round the final scores.

---

> 💡 **Important Note**
>
> TF-IDF is a feature engineering technique rather than a language model. It represents documents numerically based on word importance but cannot understand context, grammar, or semantics. Despite this limitation, it remains a strong baseline for many classical NLP tasks.

---

## Solution

### Custom Implementation

```python
import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """

    # IDF(t)=log((N+1)/(df(t)+1))+1
    # Adding +1 inside the fraction prevents division by zero.
    # Adding +1 outside the log keeps IDF positive.

    if not corpus:
        return []

    N = len(corpus)
    idf_dict = {}

    for word in set(query):
        df = sum(1 for doc in corpus if word in doc)
        idf_dict[word] = np.log((N + 1) / (df + 1)).item() + 1

    result = []

    for doc in corpus:
        row = []

        for word in query:
            tf = doc.count(word) / len(doc) if doc else 0
            row.append(round(tf * idf_dict[word], 5))

        result.append(row)

    return result
```

---

## Code Explanation

### Step 1

Check whether the corpus is empty.

```python
if not corpus:
    return []
```

This prevents unnecessary computations and handles the edge case safely.

---

### Step 2

Compute the document frequency for every unique query word.

```python
df = sum(1 for doc in corpus if word in doc)
```

Document frequency counts how many documents contain the given term.

---

### Step 3

Compute the smoothed IDF.

```python
idf_dict[word] = np.log((N + 1) / (df + 1)).item() + 1
```

The smoothing prevents division by zero and keeps the IDF value positive.

---

### Step 4

Compute the Term Frequency for every query word in every document.

```python
tf = doc.count(word) / len(doc) if doc else 0
```

TF is simply the proportion of occurrences of the word within the document.

---

### Step 5

Multiply TF and IDF.

```python
tf * idf_dict[word]
```

This produces the TF-IDF score for the current word in the current document.

---

### Step 6

Round the values.

```python
round(tf * idf_dict[word], 5)
```

The problem requires every TF-IDF score to be rounded to **five decimal places**.

---

### Step 7

Return the result.

Each row corresponds to one document, while each column corresponds to one query word.

The final output is therefore a list of TF-IDF vectors.

---

## Time & Space Complexity

| Complexity | Value            |
| ---------- | ---------------- |
| Time       | **O(Q × D × L)** |
| Space      | **O(Q + DQ)**    |

where

- $Q$ is the number of query words.
- $D$ is the number of documents.
- $L$ is the average number of words per document.
- Computing document frequencies requires scanning every document for each query term.
- Computing TF scores requires counting query terms in every document.
- The output matrix stores one TF-IDF score for every `(document, query)` pair.
