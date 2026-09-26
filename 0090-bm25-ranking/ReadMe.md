# BM25 Ranking (Medium, NLP)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: BM25](#learn-bm25)
  - [What is BM25?](#what-is-bm25)
  - [Information Retrieval](#information-retrieval)
  - [Why TF-IDF Is Not Enough](#why-tf-idf-is-not-enough)
  - [Term Frequency](#term-frequency)
  - [Term Frequency Saturation](#term-frequency-saturation)
  - [Document Length Normalization](#document-length-normalization)
  - [Inverse Document Frequency](#inverse-document-frequency)
  - [Complete BM25 Formula](#complete-bm25-formula)
  - [Parameters k1 and b](#parameters-k1-and-b)
  - [Query Scoring](#query-scoring)
  - [Example Calculation](#example-calculation)
  - [Properties](#properties)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [Alternative Implementation](#alternative-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [BM25 Ranking](https://www.deep-ml.com/problems/90)

Implement the BM25 ranking function to calculate relevance scores for a query against a collection of documents.

BM25 is a probabilistic information-retrieval ranking function related to TF-IDF.

Unlike basic TF-IDF, BM25 accounts for:

- Term frequency saturation.
- Document length normalization.
- The importance of a query term across the corpus.

Given a tokenized corpus and a tokenized query, calculate one BM25 score for every document.

The function should return the scores in the same order as the input documents.

## Example

### Input

```python id="b7k2qf"
corpus = [
    ['the', 'cat', 'sat'],
    ['the', 'dog', 'ran'],
    ['the', 'bird', 'flew']
]

query = ['the', 'cat']
```

### Output

```text id="p3m8vx"
[0.693, 0.0, 0.0]
```

### Reasoning

There are:

\(N=3\)

documents.

Every document has length:

\(dl=3\)

so the average document length is:

\(adl=3\)

The query contains two unique terms:

\(\{the,cat\}\)

The term `the` occurs in all three documents.

Therefore:

\(df(the)=3\)

Using the problem's IDF definition:

\(IDF(the)=\log\left(\frac{N+1}{df+1}\right)\)

we get:

\(IDF(the)=\log\left(\frac{4}{4}\right)=0\)

Therefore, `the` contributes nothing to any document score.

The term `cat` appears in only the first document.

Thus:

\(df(cat)=1\)

and:

\(IDF(cat)=\log\left(\frac{4}{2}\right)=\log(2)\approx0.693\)

For the first document:

\(TF(cat)=1\)

Because its length equals the average document length:

\(1-b+b\frac{dl}{adl}=1\)

With:

\(k_1=1.5\)

the term-frequency component becomes:

\(\frac{TF(k_1+1)}{TF+k_1}=1\)

Therefore:

\(BM25(cat)\approx0.693\)

The other documents do not contain `cat`, so their scores are:

\(0\)

Hence:

```text
[0.693, 0.0, 0.0]
```

## Learn: BM25

### What is BM25?

BM25 is a ranking function used in information retrieval.

Its purpose is to estimate how relevant a document is to a search query.

Given:

- A collection of documents.
- A query.
- A term appearing in both.

BM25 assigns a numerical score representing the contribution of that term to the document's relevance.

The scores can then be used to rank documents.

Higher BM25 scores generally indicate stronger matching under the model.

BM25 is widely used in classical search systems and remains an important baseline even in modern retrieval systems.

### Information Retrieval

Information retrieval is the problem of finding relevant documents for a user's information need.

For example, a user may search:

```text
machine learning optimization
```

A search system must determine which documents are most relevant.

A simple approach might count how often the query terms occur.

However, raw term counts are not sufficient.

A term occurring many times in a document does not necessarily make that document proportionally more relevant.

A common word appearing in almost every document is also less useful for distinguishing relevant documents.

BM25 addresses both of these issues.

### Ranking vs Classification

BM25 does not classify documents into fixed categories.

Instead, it assigns a relevance score.

For documents:

\(D_1,D_2,\ldots,D_N\)

BM25 produces:

\(S(D_1,Q),S(D_2,Q),\ldots,S(D_N,Q)\)

The documents can then be sorted by score.

This makes BM25 a ranking function.

### Why TF-IDF Is Not Enough

TF-IDF combines:

- Term frequency.
- Inverse document frequency.

The basic idea is useful, but raw term frequency has an important limitation.

Suppose a document contains a query term once:

\(TF=1\)

Now suppose another document contains it 100 times:

\(TF=100\)

A raw TF-based system can assign a much larger contribution to the second document.

However, the difference between seeing a useful term once and seeing it several times is usually more significant than the difference between seeing it 50 and 100 times.

The relevance contribution should therefore saturate.

BM25 introduces a nonlinear TF transformation to achieve this.

### Term Frequency

Term frequency is the number of times a term occurs in a document.

For a term $t$ and document $d$:

\(TF(t,d)=\text{count of }t\text{ in }d\)

For example:

```text
document = ["cat", "dog", "cat", "bird"]
```

For the term `cat`:

\(TF(cat,d)=2\)

For `dog`:

\(TF(dog,d)=1\)

For `fish`:

\(TF(fish,d)=0\)

In the implementation, this is obtained with:

```python id="h4n8sv"
tf = doc.count(word)
```

### Term Frequency Saturation

BM25 does not use raw term frequency directly.

Instead, it uses:

\(\frac{TF(k_1+1)}{TF+k_1}\)

This function increases as $TF$ increases, but it approaches a limit.

For large $TF$:

\(\lim\_{TF\rightarrow\infty}\frac{TF(k_1+1)}{TF+k_1}=k_1+1\)

Therefore, repeatedly adding the same word produces diminishing returns.

### Intuition Behind Saturation

Consider a term with:

\(k_1=1.5\)

For:

\(TF=1\)

the TF component is:

\(\frac{1(2.5)}{1+1.5}=1\)

For:

\(TF=2\)

it becomes:

\(\frac{2(2.5)}{2+1.5}\approx1.429\)

For:

\(TF=10\)

it becomes:

\(\frac{10(2.5)}{10+1.5}\approx2.174\)

The score continues increasing, but the increase becomes progressively smaller.

This is the saturation behavior that distinguishes BM25 from a raw term-frequency model.

### Document Length Normalization

A long document naturally has more opportunities for a query term to appear.

Suppose one document contains 100 words and another contains 10,000 words.

Finding a term five times in the 10,000-word document should not necessarily be considered more relevant than finding it five times in the short document.

BM25 accounts for this using document length normalization.

The normalization term is:

\(1-b+b\frac{dl}{adl}\)

where:

- $dl$ is the document length.
- $adl$ is the average document length.
- $b$ controls the strength of normalization.

### Average Document Length

For $N$ documents:

\(adl=\frac{1}{N}\sum\_{d=1}^{N}|d|\)

where $|d|$ is the number of tokens in document $d$.

For example, if document lengths are:

\(3,5,7\)

then:

\(adl=\frac{3+5+7}{3}=5\)

The average length provides a reference point for determining whether a document is short or long.

### Length Normalization Behavior

The normalization term is:

\(1-b+b\frac{dl}{adl}\)

If:

\(dl=adl\)

then:

\(1-b+b=1\)

So an average-length document receives no length adjustment.

If:

\(dl>adl\)

the denominator becomes larger.

This reduces the TF contribution.

If:

\(dl<adl\)

the denominator becomes smaller.

This increases the TF contribution.

Thus, longer documents receive a stronger penalty while shorter documents receive a relative advantage.

### Parameter b

The parameter $b$ controls document-length normalization.

The usual range is:

\(0\leq b\leq1\)

When:

\(b=0\)

the length normalization disappears:

\(1-b+b\frac{dl}{adl}=1\)

Therefore, document length has no effect.

When:

\(b=1\)

length normalization is applied fully according to the document-to-average-length ratio.

A common default is:

\(b=0.75\)

which is also the value used in this implementation.

### Parameter k1

The parameter $k_1$ controls the strength of term-frequency saturation.

The implementation uses:

\(k_1=1.5\)

Increasing $k_1$ changes how quickly the TF component approaches saturation.

The exact effect is easiest to understand from:

\(\frac{TF(k_1+1)}{TF+k_1}\)

BM25 implementations commonly use values around $1$ to $2$ as practical defaults, although the best value depends on the retrieval system and dataset.

### Inverse Document Frequency

Not every word is equally useful for retrieval.

A term appearing in nearly every document provides little information about which document is relevant.

A rare term is more discriminative.

BM25 therefore assigns a higher weight to terms with lower document frequency.

The problem uses:

\(IDF(t)=\log\left(\frac{N+1}{df(t)+1}\right)\)

where:

- $N$ is the number of documents.
- $df(t)$ is the number of documents containing term $t$.

### Document Frequency

Document frequency counts documents containing a term.

It is different from total term frequency.

Suppose:

```text
D1 = ["cat", "cat", "cat"]
D2 = ["dog"]
D3 = ["bird"]
```

For `cat`:

\(TF(cat,D_1)=3\)

but:

\(df(cat)=1\)

because only one document contains the term.

Document frequency asks:

> In how many documents does this term appear?

Term frequency asks:

> How many times does this term appear in this particular document?

These are different quantities.

### IDF Behavior

Using:

\(IDF(t)=\log\left(\frac{N+1}{df(t)+1}\right)\)

a term appearing in every document has:

\(df=N\)

and therefore:

\(IDF=\log(1)=0\)

Such a term contributes no discriminative value.

A term appearing in fewer documents receives a positive IDF.

For example, if:

\(N=100\)

and:

\(df=1\)

then:

\(IDF=\log\left(\frac{101}{2}\right)\)

which is substantially larger than the IDF of a term appearing in most documents.

### Complete BM25 Formula

For one term $t$ in document $d$, the BM25 contribution used by this problem is:

\(BM25(t,d)=IDF(t)\frac{TF(t,d)(k_1+1)}{TF(t,d)+k_1\left(1-b+b\frac{dl}{adl}\right)}\)

For a query containing multiple unique terms, the document score is the sum of the contributions:

\(Score(d,Q)=\sum\_{t\in Q}BM25(t,d)\)

The implementation iterates over:

```python id="q7m2ka"
set(query)
```

so repeated occurrences of the same query term do not create multiple copies of the same BM25 contribution.

### Why Sum the Query Terms?

A query normally contains multiple terms.

Suppose:

```text
query = ["machine", "learning"]
```

A document might contain:

- `machine` but not `learning`.
- `learning` but not `machine`.
- Both.
- Neither.

Each query term contributes independently according to its BM25 score.

Therefore:

\(Score(d,Q)=BM25(machine,d)+BM25(learning,d)\)

A document matching multiple informative query terms can receive a higher total score.

### Query Term Deduplication

The implementation uses:

```python id="x3k8vp"
for word in set(query):
```

This means a repeated query term is processed only once.

For example:

```python
query = ["cat", "cat", "dog"]
```

becomes conceptually:

```text
{"cat", "dog"}
```

This is a reasonable interpretation for the problem's implementation.

Some BM25 variants or query-processing pipelines may incorporate query term frequency or query expansion, but the standard basic formulation commonly treats the query as a set of terms.

### Complete Scoring Process

For each unique query term:

1. Count how many documents contain the term.
2. Compute its IDF.
3. For each document, calculate term frequency.
4. Calculate the document-length normalization factor.
5. Calculate the saturated TF component.
6. Multiply by IDF.
7. Add the contribution to the document's total score.

The final result is one score per document.

### Example Calculation

Consider:

```text
D1 = ["the", "cat", "sat"]
D2 = ["the", "dog", "ran"]
D3 = ["the", "bird", "flew"]

Q = ["the", "cat"]
```

There are:

\(N=3\)

documents.

Every document has length:

\(dl=3\)

Therefore:

\(adl=3\)

### Term: "the"

The term appears in all documents:

\(df(the)=3\)

Thus:

\(IDF(the)=\log\left(\frac{3+1}{3+1}\right)=0\)

Therefore:

\(BM25(the,d)=0\)

for every document.

### Term: "cat"

The term occurs only in $D_1$:

\(df(cat)=1\)

Therefore:

\(IDF(cat)=\log\left(\frac{4}{2}\right)=\log(2)\approx0.693\)

For $D_1$:

\(TF(cat,D_1)=1\)

The length factor is:

\(1-0.75+0.75\frac{3}{3}=1\)

The TF component becomes:

\(\frac{1(1.5+1)}{1+1.5(1)}=1\)

Therefore:

\(BM25(cat,D_1)=0.693\)

For $D_2$:

\(TF(cat,D_2)=0\)

so:

\(BM25(cat,D_2)=0\)

The same applies to $D_3$.

Therefore:

\(Scores=[0.693,0,0]\)

### Zero Term Frequency

If a document does not contain a query term:

\(TF=0\)

The BM25 TF component becomes:

\(\frac{0(k_1+1)}{0+k_1(...)}=0\)

Therefore, the term contributes nothing.

This is why the implementation can safely calculate the formula without a special case for missing query terms, assuming valid positive document lengths and parameters.

### Relationship Between TF and IDF

BM25 combines two distinct ideas.

Term frequency asks:

> How strongly does this document contain the query term?

IDF asks:

> How informative is this query term across the corpus?

A term contributes strongly when:

- It appears in the document.
- It occurs enough times to matter.
- It is relatively rare across the corpus.
- Its occurrence is meaningful relative to document length.

This combination makes BM25 effective for document ranking.

### BM25 vs TF-IDF

TF-IDF generally uses a multiplicative structure involving TF and IDF.

A simplified TF-IDF formulation might be:

\(TFIDF(t,d)=TF(t,d)\times IDF(t)\)

BM25 replaces raw TF with a saturating function and adds document-length normalization.

The central BM25 term is:

\(\frac{TF(k_1+1)}{TF+k_1(1-b+b\frac{dl}{adl})}\)

Therefore BM25 can be viewed as a more controlled ranking function rather than simply multiplying raw term frequency by IDF.

### BM25 Is Not a Probability

Despite its probabilistic origins, the numerical BM25 score itself is not a probability.

For example:

\(Score(d,Q)=4.2\)

does not mean there is a $420%$ probability of relevance.

The score is a ranking signal.

Documents can be ordered by their scores.

### Ranking Documents

Suppose BM25 produces:

```text
[4.2, 1.8, 0.4, 3.1]
```

The corresponding ranking from highest to lowest score would be:

```text
document 1
document 4
document 2
document 3
```

The score itself is meaningful primarily relative to other documents scored for the same query.

### Applications

BM25 is widely used in:

- Search engines.
- Document retrieval.
- Enterprise search.
- Question-answering retrieval.
- Recommendation systems.
- NLP pipelines.
- Legal document search.
- Academic search.
- E-commerce search.
- Retrieval-Augmented Generation.

### BM25 in RAG

BM25 is especially useful in retrieval-augmented generation systems.

A knowledge base can contain thousands or millions of text chunks.

Given a user query, BM25 can retrieve documents containing important lexical matches.

A generative model can then receive the retrieved documents as context.

BM25 is particularly useful when exact terms matter.

For example, a query containing a product identifier, API name, error code, or specific technical term may benefit from lexical retrieval.

### Lexical vs Semantic Retrieval

BM25 is a lexical retrieval method.

It primarily cares about matching words.

A semantic embedding system can retrieve documents with similar meanings even when the exact words differ.

For example:

```text
query: "car stopped working"
```

A semantic retriever might match:

```text
"vehicle engine failure"
```

even though the words differ significantly.

BM25 may not consider the two texts strongly related unless useful lexical overlap exists.

Modern retrieval systems often combine lexical and semantic retrieval.

### Important Parameters

The most important BM25 parameters in this implementation are:

\(k_1=1.5\)

and:

\(b=0.75\)

$k_1$ controls term-frequency saturation.

$b$ controls document-length normalization.

The IDF formula is also important because different BM25 variants use slightly different IDF definitions.

### IDF Variant Matters

There is not one universally identical BM25 implementation.

Different libraries and papers can use different IDF formulas or additional smoothing.

This problem explicitly specifies:

\(IDF=\log\left(\frac{N+1}{df+1}\right)\)

Therefore, the implementation should follow that formula rather than silently replacing it with another variant.

> 💡 **Important Note**
>
> BM25 is a family of closely related scoring formulations. When implementing it for a coding problem, follow the exact IDF and normalization convention specified by the problem. Different search libraries may produce different numerical scores for the same documents.

### Applications

BM25 is useful when a retrieval system needs fast and interpretable lexical matching.

Typical uses include:

- Search result ranking.
- Candidate generation.
- Document retrieval.
- FAQ retrieval.
- Knowledge-base search.
- RAG preprocessing.
- Hybrid retrieval systems.
- Text matching.

## Solutions

### Custom Implementation

```python id="s6v2kd"
import numpy as np

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus:
        return []

    N = len(corpus)
    avg_doc_len = sum(len(doc) for doc in corpus) / N

    idf_dict = {}

    for word in set(query):
        df = sum(1 for doc in corpus if word in doc)
        idf_dict[word] = np.log((N + 1) / (df + 1)).item()

    scores = []

    for doc in corpus:
        doc_score = 0

        for word in set(query):
            tf = doc.count(word)

            bm25 = idf_dict[word] * (
                (tf * (k1 + 1))
                / (tf + k1 * (1 - b + b * (len(doc) / avg_doc_len)))
            )

            doc_score += bm25

        scores.append(round(doc_score, 3))

    return scores
```

### Alternative Implementation

The same logic can be written without NumPy because the only NumPy operation required by the supplied solution is the logarithm.

```python id="n8r4yc"
import math

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus:
        return []

    N = len(corpus)
    avg_doc_len = sum(len(doc) for doc in corpus) / N
    unique_query = set(query)

    idf = {}

    for word in unique_query:
        df = sum(word in doc for doc in corpus)
        idf[word] = math.log((N + 1) / (df + 1))

    scores = []

    for doc in corpus:
        score = 0.0
        doc_len = len(doc)

        for word in unique_query:
            tf = doc.count(word)

            denominator = tf + k1 * (
                1 - b + b * (doc_len / avg_doc_len)
            )

            score += idf[word] * (
                tf * (k1 + 1)
            ) / denominator

        scores.append(round(score, 3))

    return scores
```

The second version makes the mathematical structure slightly easier to inspect.

## Code Explanation

### 1. Handle an Empty Corpus

```python id="q2f6mn"
if not corpus:
    return []
```

There are no documents to rank.

Therefore, the function immediately returns an empty list.

This also avoids division by zero when calculating average document length.

### 2. Count the Documents

```python id="v4k8rs"
N = len(corpus)
```

BM25 needs the total number of documents because IDF depends on corpus size.

The variable:

\(N\)

represents the number of documents.

### 3. Compute Average Document Length

```python id="a5n9qx"
avg_doc_len = sum(len(doc) for doc in corpus) / N
```

This calculates:

\(adl=\frac{\sum_d|d|}{N}\)

The average length is used to normalize the effect of document size.

### 4. Create the IDF Dictionary

```python id="m7p3cz"
idf_dict = {}
```

The dictionary stores the IDF value for every unique query term.

There is no need to calculate IDF for words that never appear in the query.

### 5. Iterate Over Unique Query Terms

```python id="t8x5vk"
for word in set(query):
```

Using a set prevents duplicate query terms from being processed repeatedly.

If:

```text
query = ["cat", "cat", "dog"]
```

only `cat` and `dog` are evaluated.

### 6. Calculate Document Frequency

```python id="r3w9fd"
df = sum(1 for doc in corpus if word in doc)
```

This counts the number of documents containing the query term.

Importantly, it counts documents rather than occurrences.

For example:

```text
D1 = ["cat", "cat", "cat"]
D2 = ["dog"]
```

gives:

\(df(cat)=1\)

not $3$.

### 7. Calculate IDF

```python id="y6k2pm"
idf_dict[word] = np.log((N + 1) / (df + 1)).item()
```

This implements the problem's IDF formula:

\(IDF(t)=\log\left(\frac{N+1}{df(t)+1}\right)\)

The `.item()` converts the NumPy scalar into a normal Python scalar.

### 8. Initialize Document Scores

```python id="c4n7hx"
scores = []
```

The list will eventually contain one BM25 score for each document.

The ordering matches the ordering of `corpus`.

### 9. Score Each Document

```python id="p8r5vs"
for doc in corpus:
    doc_score = 0
```

Each document receives its own accumulated score.

The score starts at zero.

### 10. Compute Term Frequency

```python id="u2m9qa"
tf = doc.count(word)
```

This counts the number of occurrences of the query term in the current document.

This is:

\(TF(t,d)\)

Term frequency is later transformed by the BM25 saturation function.

### 11. Compute the BM25 Term Score

```python id="w6q3kr"
bm25 = idf_dict[word] * (
    (tf * (k1 + 1))
    / (tf + k1 * (1 - b + b * (len(doc) / avg_doc_len)))
)
```

This directly implements:

\(BM25(t,d)=IDF(t)\frac{TF(t,d)(k_1+1)}{TF(t,d)+k_1(1-b+b\frac{dl}{adl})}\)

The expression contains three important components.

#### IDF

```python
idf_dict[word]
```

Measures how informative the term is across the corpus.

#### Saturated TF

```python
(tf * (k1 + 1))
```

appears in the numerator of the BM25 TF component.

#### Length-Normalized Denominator

```python
tf + k1 * (1 - b + b * (len(doc) / avg_doc_len))
```

controls both TF saturation and document-length normalization.

### 12. Accumulate Query-Term Contributions

```python id="z4p7hx"
doc_score += bm25
```

BM25 scores for all unique query terms are added together.

Therefore:

\(Score(d,Q)=\sum\_{t\in Q}BM25(t,d)\)

### 13. Round the Score

```python id="e9k3wc"
scores.append(round(doc_score, 3))
```

The result is rounded to three decimal places.

This matches the expected output format in the problem.

Rounding is primarily a presentation choice.

It does not change the underlying BM25 calculation before rounding.

### 14. Return All Scores

```python id="f2m8vs"
return scores
```

The returned list contains one score per corpus document.

The original document ordering is preserved.
