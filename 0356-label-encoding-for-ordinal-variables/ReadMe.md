# Label Encoding for Ordinal Variables (Easy, Data Preprocessing)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Label Encoding for Ordinal Variables](#learn-label-encoding-for-ordinal-variables)
  - [Categorical Variables](#categorical-variables)
  - [Nominal vs Ordinal Variables](#nominal-vs-ordinal-variables)
  - [Ordinal Variables](#ordinal-variables)
  - [Nominal Variables](#nominal-variables)
  - [Why Order Matters](#why-order-matters)
  - [Ordinal Encoding](#ordinal-encoding)
  - [Encoding Function](#encoding-function)
  - [Unknown Categories](#unknown-categories)
  - [Advantages](#advantages)
  - [When to Use](#when-to-use)
  - [When Not to Use](#when-not-to-use)
  - [Ordinal Encoding vs One-Hot Encoding](#ordinal-encoding-vs-one-hot-encoding)
  - [Equal Spacing Assumption](#equal-spacing-assumption)
  - [Tree-Based Models](#tree-based-models)
  - [Linear Models](#linear-models)
  - [Data Leakage](#data-leakage)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [Scikit-learn Equivalent](#scikit-learn-equivalent)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Label Encoding for Ordinal Variables](https://www.deep-ml.com/problems/356)

Implement a function that performs label encoding for an **ordinal
categorical variable** while preserving the natural ordering between
categories.

An ordinal variable contains categories with a meaningful ranking.

For example:

```text
high school < bachelor < master < phd
```

or:

```text
poor < fair < good < excellent
```

The input contains two lists:

- `values`: categorical values that need to be encoded
- `order`: categories arranged from lowest rank to highest rank

The first category in `order` receives integer `0`.

The second receives integer `1`.

The third receives integer `2`.

This continues until every category has a numerical rank.

For example:

```text id="7b8k5u"
order = ["small", "medium", "large"]
```

defines the mapping:

```text id="b4m8t2"
small  -> 0
medium -> 1
large  -> 2
```

If a value does not exist in `order`, the function must return `-1`
for that value.

An empty `values` list must produce an empty list.

---

## Example

### Input

```python id="y5u2x7"
values = ["medium", "small", "large", "small"]
order = ["small", "medium", "large"]
```

### Output

```text id="m0b3cx"
[1, 0, 2, 0]
```

### Reasoning

The ordering list defines:

```text id="7hj4f2"
small  -> 0
medium -> 1
large  -> 2
```

The input is:

```text id="5n9a2e"
["medium", "small", "large", "small"]
```

Therefore:

```text id="5k1w8q"
medium -> 1
small  -> 0
large  -> 2
small  -> 0
```

The final encoded sequence is:

```text id="q3z7nm"
[1, 0, 2, 0]
```

---

## Example with Unknown Categories

Suppose:

```python id="1s8v3a"
values = ["small", "large", "extra-large"]
order = ["small", "medium", "large"]
```

The mapping is:

```text id="p7y2k4"
small  -> 0
medium -> 1
large  -> 2
```

`extra-large` is not present in the ordering.

Therefore:

```text id="4k0z9x"
[0, 2, -1]
```

The `-1` explicitly represents an unknown category.

---

# Learn: Label Encoding for Ordinal Variables

## Categorical Variables

A categorical variable contains values representing categories rather
than continuous numerical measurements.

Examples include:

```text
Color
Education level
Product size
Satisfaction level
Country
Payment type
```

A dataset might contain:

```text
Education
---------
bachelor
master
phd
high school
```

Machine learning algorithms generally operate on numerical
representations.

Therefore, categorical features often need to be transformed before
being passed to a model.

The appropriate transformation depends on whether the categories have
an inherent ordering.

---

## Nominal vs Ordinal Variables

The distinction between **nominal** and **ordinal** categorical
variables is fundamental.

### Nominal

Nominal categories have no meaningful ranking.

Examples:

```text
red
blue
green
```

There is no natural statement that:

\(red<blue<green\)

The category labels are simply different.

Other examples include:

```text
country
browser
operating system
payment method
```

---

### Ordinal

Ordinal categories have a meaningful ordering.

Examples:

```text
small < medium < large
```

and:

```text
poor < fair < good < excellent
```

The ordering contains useful information.

Therefore, encoding should preserve that ordering.

---

## Ordinal Variables

An ordinal variable can be represented as an ordered collection:

\(C=\{c*0,c_1,\ldots,c*{n-1}\}\)

where the index represents the rank.

For example:

\(C=\{small,medium,large\}\)

with:

\(small<medium<large\)

The corresponding encoding is:

\(f(small)=0\)

\(f(medium)=1\)

\(f(large)=2\)

The numerical representation preserves the ranking.

---

## Nominal Variables

For nominal categories, assigning arbitrary integers can create a
misleading ordering.

Suppose:

```text
red -> 0
blue -> 1
green -> 2
```

A model may interpret:

\(0<1<2\)

as meaningful information.

But there is no inherent relationship such as:

\(red<blue<green\)

Therefore, simple integer encoding is generally inappropriate for
nominal categories.

One-hot encoding is often more appropriate.

---

## Why Order Matters

Consider the ordinal variable:

```text
small < medium < large
```

Encoding it as:

```text
small  -> 0
medium -> 1
large  -> 2
```

preserves the ranking:

\(0<1<2\)

A model can therefore distinguish that `large` represents a higher
ordinal category than `medium`.

If the categories were instead encoded randomly:

```text
small  -> 2
medium -> 0
large  -> 1
```

the numerical representation would no longer preserve the original
ordering.

The encoding must therefore be based on domain knowledge rather than
alphabetical order or arbitrary category discovery.

---

## Ordinal Encoding

Ordinal encoding maps each category to an integer representing its rank.

Given:

```text id="4s0v2n"
order = ["poor", "fair", "good", "excellent"]
```

the mapping becomes:

```text id="7z1m8x"
poor      -> 0
fair      -> 1
good      -> 2
excellent -> 3
```

The transformation is:

\(f(c_i)=i\)

where $c_i$ is the category at position $i$ in the provided ordering.

---

## Encoding Function

Suppose:

\(C=\{c*0,c_1,\ldots,c*{n-1}\}\)

is the ordered category set.

The encoding function is:

\(f(c_i)=i\)

For an input sequence:

\(V=[v_1,v_2,\ldots,v_m]\)

the encoded sequence becomes:

\(E=[f(v_1),f(v_2),\ldots,f(v_m)]\)

For example:

```text
order = ["low", "medium", "high"]
values = ["high", "low", "medium"]
```

produces:

```text
[2, 0, 1]
```

---

## Mapping Representation

The most efficient implementation for repeated lookups is a dictionary.

For:

```python id="j2h7c4"
order = ["small", "medium", "large"]
```

construct:

```python id="s8x3d1"
{
    "small": 0,
    "medium": 1,
    "large": 2
}
```

Then each input value can be looked up directly.

This is preferable to searching the `order` list for every value.

---

## Dictionary Construction

Python provides `enumerate` for obtaining both the position and the
category:

```python id="2z5x7a"
mapping = {value: index for index, value in enumerate(order)}
```

For:

```text id="d9r1c6"
order = ["small", "medium", "large"]
```

the dictionary becomes:

```text id="r5x0n3"
{
    "small": 0,
    "medium": 1,
    "large": 2
}
```

The index itself represents the ordinal rank.

---

## Unknown Categories

Real datasets may contain categories that were not present in the
predefined ordering.

For example:

```text id="p4s7j2"
order = ["small", "medium", "large"]

value = "extra-large"
```

Since:

```text id="m6q1w9"
"extra-large" not in order
```

the function returns:

```text id="8z3k1a"
-1
```

This explicitly indicates that the category is unknown.

The mapping function can therefore be defined as:

$$
f(v)=
\begin{cases}
i & \text{if }v=c_i\\
-1 & \text{if }v\notin C
\end{cases}$$

---

## Why Use -1?

The problem specifies `-1` as the representation for an unknown
category.

It provides a value outside the valid rank range:

$$0\leq f(v)\leq n-1$$

for known categories.

Therefore:

$$-1$$

can be interpreted as a special unknown or invalid category marker.

In a production machine learning pipeline, however, the handling of
unknown categories depends on the model and preprocessing framework.

Possible alternatives include:

- a dedicated unknown-category index
- a special string category
- an error
- an ignored category
- a reserved integer code

---

## Advantages of Ordinal Encoding

### Preserves Ordering

The most important advantage is that the natural ranking is explicitly
represented.

For example:

```text
poor -> 0
fair -> 1
good -> 2
excellent -> 3
```

preserves:

$$poor<fair<good<excellent$$

---

### Memory Efficient

An ordinal feature represented using one integer per observation is
compact.

Instead of creating several binary columns, one numerical feature is
used.

For $n$ ordered categories, one value is sufficient:

$$x\in\{0,1,\ldots,n-1\}$$

---

### Simple

Ordinal encoding is straightforward to understand and implement.

The transformation is deterministic once the category ordering is
specified.

---

## When to Use

Ordinal encoding is appropriate when:

- categories have a meaningful order
- the ordering is known
- the rank itself contains useful information
- the model can appropriately use the encoded representation

Examples include:

```text
Education:
high school < bachelor < master < phd
```

```text
Size:
small < medium < large
```

```text
Satisfaction:
poor < fair < good < excellent
```

```text
Severity:
low < moderate < high < critical
```

---

## When Not to Use

Do not use ordinal encoding simply because a categorical feature needs
numbers.

The categories must have a meaningful order.

For example:

```text
red
blue
green
```

should not normally become:

```text
red -> 0
blue -> 1
green -> 2
```

because the assigned numerical relationship is arbitrary.

For such nominal variables, one-hot encoding or another categorical
representation is usually more appropriate.

---

## Ordinal Encoding vs One-Hot Encoding

Consider:

```text
["small", "medium", "large"]
```

Ordinal encoding produces:

```text
[0, 1, 2]
```

One-hot encoding produces three binary features:

```text
small   medium   large
1       0        0
0       1        0
0       0        1
```

Ordinal encoding preserves ranking.

One-hot encoding does not impose a ranking.

Therefore:

```text
Ordinal:
small < medium < large
```

is represented naturally by ordinal encoding.

For nominal data:

```text
red
blue
green
```

one-hot encoding avoids creating an artificial ranking.

---

## Equal Spacing Assumption

There is an important subtlety with ordinal encoding.

Consider:

```text
poor -> 0
fair -> 1
good -> 2
excellent -> 3
```

The encoding preserves the order.

But it also places the categories at equally spaced numerical positions.

The difference between:

$$0\rightarrow1$$

is numerically the same as:

$$2\rightarrow3$$

This does **not necessarily mean** that the real-world difference
between `poor` and `fair` is equal to the difference between `good` and
`excellent`.

This distinction is important.

Ordinal encoding guarantees ranking, but the integer distances may not
have a meaningful quantitative interpretation.

---

## Linear Models

For linear models, integer encoding can introduce assumptions about
spacing.

Suppose:

```text
poor -> 0
fair -> 1
good -> 2
excellent -> 3
```

A linear model may use:

$$\hat y=\beta_0+\beta_1x$$

This assumes a constant numerical effect for every one-unit increase in
the encoded feature.

That means moving from:

```text
poor -> fair
```

is treated numerically like:

```text
good -> excellent
```

The assumption may or may not be appropriate.

If the ordinal levels have uneven effects, alternative encodings or
modeling strategies may be preferable.

---

## Tree-Based Models

Decision trees can use threshold splits such as:

$$x<2$$

For ordinal encoding:

```text
0 -> poor
1 -> fair
2 -> good
3 -> excellent
```

the split:

$$x<2$$

corresponds to:

```text
poor
fair
```

versus:

```text
good
excellent
```

This can naturally exploit the ordering.

Tree-based models therefore often work effectively with ordinal integer
representations.

However, model-specific behavior should still be evaluated rather than
assuming ordinal encoding is universally optimal.

---

## Encoding Based on Domain Knowledge

The ordering should come from the semantics of the variable.

Suppose:

```text
education = ["high school", "bachelor", "master", "phd"]
```

The order is domain-defined.

It should not be generated simply by:

```python id="f1v5j0"
sorted(categories)
```

because alphabetical ordering is not necessarily the natural ranking.

For example, alphabetical sorting could produce:

```text
bachelor
high school
master
phd
```

which is not the intended ordinal relationship.

The explicit `order` argument solves this problem.

---

## Data Leakage

The category ordering itself is usually based on domain knowledge, so
it may be defined before training.

However, if categories or encoding mappings are learned from training
data, the mapping should be fitted using the training set and then
applied consistently to validation and test data.

The essential principle is:

```text id="n0p7y2"
Fit mapping
     |
     v
Training data
     |
     +----> Validation data
     |
     +----> Test data
```

The mapping should not change between datasets.

Otherwise, the same numerical value could represent different
categories across splits.

---

## Consistent Mapping

Suppose training data uses:

```text
small  -> 0
medium -> 1
large  -> 2
```

The validation and test data must use the same mapping.

Do not independently generate:

```text
validation:
medium -> 0
large  -> 1
small  -> 2
```

because the numerical representation would no longer have the same
meaning.

A preprocessing pipeline should therefore treat the ordering or
category mapping as part of the fitted transformation.

---

## Unknown Categories in Production

Unknown categories are particularly important when a model encounters
new data.

Suppose training supports:

```text
small
medium
large
```

but production data contains:

```text
extra-large
```

A robust pipeline needs a defined policy.

This problem uses:

```text
extra-large -> -1
```

In production systems, a dedicated unknown category is often preferable
to allowing the pipeline to fail unexpectedly.

---

## Unknown Does Not Mean Lowest Rank

The value:

```text
-1
```

should not necessarily be interpreted semantically as "smaller than
small."

It is a special code indicating that the category is not known.

For:

```text
small -> 0
medium -> 1
large -> 2
unknown -> -1
```

the numerical ordering:

$$-1<0<1<2$$

does not necessarily represent the real-world ordering.

This matters when feeding encoded values into models.

A model may interpret `-1` numerically unless the preprocessing design
accounts for it.

---

## Applications

Ordinal encoding is useful for features such as:

- education level
- customer satisfaction
- product size
- severity level
- quality rating
- risk category
- experience level
- service tier
- agreement level
- socioeconomic categories

A typical preprocessing pipeline can contain:

```text
Raw categorical feature
          |
          v
Determine variable type
          |
       +--+--+
       |     |
   Nominal  Ordinal
       |     |
       v     v
 One-Hot  Ordinal Encoding
```

---

> 💡 **Important Note**
>
> The critical question is not "Can these categories be converted into
> integers?" Almost every categorical variable can be assigned integers.
> The important question is "Does the numerical ordering represent a
> real semantic ordering?" If the answer is no, simple ordinal encoding
> can introduce misleading structure.

---

# Solutions

The problem has a particularly clean implementation.

The key idea is to construct a dictionary mapping each category to its
rank.

For:

```text
order = ["small", "medium", "large"]
```

construct:

```text
{
    "small": 0,
    "medium": 1,
    "large": 2
}
```

Then perform a dictionary lookup for every value.

Unknown values use `-1`.

---

## Custom Implementation

```python id="c7p4w2"
def label_encode_ordinal(values: list, order: list) -> list:
    mapping = {value: index for index, value in enumerate(order)}
    return [mapping.get(value, -1) for value in values]
```

This is the user's supplied solution.

It is already optimal for the intended algorithm.

---

## Scikit-learn Equivalent

Scikit-learn provides ordinal encoding functionality through
`OrdinalEncoder`.

A typical usage is:

```python id="z6m2q8"
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

order = ["small", "medium", "large"]

encoder = OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)

values = np.array([
    ["medium"],
    ["small"],
    ["large"],
    ["small"]
])

encoded = encoder.fit_transform(values)
```

The resulting values are floating-point representations:

```text
[[1.]
 [0.]
 [2.]
 [0.]]
```

The problem itself requires a Python list of integers, so the custom
implementation is a more direct match for the requested API.

---

# Code Explanation

## Step 1: Build the Mapping

The core line is:

```python id="r8x1m3"
mapping = {value: index for index, value in enumerate(order)}
```

`enumerate(order)` produces pairs of:

```text
(index, value)
```

For:

```text id="y8k4p1"
order = ["small", "medium", "large"]
```

it produces conceptually:

```text
(0, "small")
(1, "medium")
(2, "large")
```

The dictionary comprehension reverses that into:

```text
"small"  -> 0
"medium" -> 1
"large"  -> 2
```

---

## Step 2: Encode Each Value

The return expression is:

```python id="g5r9z2"
return [mapping.get(value, -1) for value in values]
```

This iterates through the input values.

For every value, it performs a dictionary lookup.

For example:

```text id="p7q3s6"
value = "medium"
```

gives:

```text
mapping["medium"] = 1
```

Therefore:

```text id="j2v6n8"
"medium" -> 1
```

---

## Step 3: Handle Unknown Categories

The dictionary's `get` method accepts a default value:

```python id="z0c5y7"
mapping.get(value, -1)
```

If the key exists:

```text id="c2w9k4"
mapping.get("large", -1) -> 2
```

If the key does not exist:

```text id="m8f1r5"
mapping.get("extra-large", -1) -> -1
```

This directly implements the problem's unknown-category requirement.

---

## Step 4: Preserve Input Order

The function does not sort `values`.

It processes them in their original order.

For:

```text id="h3n7v0"
values = ["large", "small", "medium"]
```

the result is:

```text id="s5j2k8"
[2, 0, 1]
```

The output position corresponds directly to the input position.

Therefore:

$$E_i=f(V_i)$$

---

## Empty Input

If:

```python id="w4k8p2"
values = []
```

the list comprehension has no elements to process.

Therefore:

```text id="g1r6y9"
[]
```

is returned automatically.

No special branch is necessary.

---

## Duplicate Values

Repeated categories are encoded independently.

For:

```text id="c9m3x7"
values = ["small", "small", "large", "small"]
```

the result is:

```text id="q4z8b1"
[0, 0, 2, 0]
```

The mapping is created only once.

The same category always receives the same rank.

---

## Why a Dictionary Is Better Than List Search

A naive implementation might do:

```python id="k7p2n4"
[order.index(value) for value in values]
```

This searches through `order` for every input value.

If there are:

- $m$ input values
- $n$ categories

the worst-case lookup cost is:

$$O(n)$$

for each value.

Therefore:

$$O(mn)$$

overall in the worst case.

The dictionary approach builds the mapping once and performs average
constant-time lookups.

This reduces the expected total time to:

$$O(n+m)$$

which is substantially better when the input contains many values.

---

## Dictionary Lookup Complexity

Python dictionaries use hash tables.

Under normal conditions:

$$T_{lookup}=O(1)$$

on average.

Therefore:

```text id="v6r2x8"
Build mapping -> O(n)
Encode values -> O(m)
```

giving:

$$O(n+m)$$

expected time.

---

## Why `enumerate` Is Useful

Without `enumerate`, one might manually track the index:

```python id="b4n9q1"
index = 0
for value in order:
    ...
    index += 1
```

`enumerate` provides the index directly:

```python id="f8x3m6"
for index, value in enumerate(order):
```

This is both simpler and less error-prone.

---

# Worked Example: Education

Suppose:

```python id="y7m2k5"
order = ["high school", "bachelor", "master", "phd"]
```

The mapping becomes:

```text id="z4r8c1"
high school -> 0
bachelor    -> 1
master      -> 2
phd         -> 3
```

For:

```text id="k6q1v9"
values = ["master", "phd", "high school"]
```

the result is:

```text id="m3t7x2"
[2, 3, 0]
```

The numerical representation preserves the educational ordering.

---

# Worked Example: Satisfaction

Suppose:

```python id="c8n4p6"
order = ["poor", "fair", "good", "excellent"]
```

Then:

```text id="s2v7j1"
poor      -> 0
fair      -> 1
good      -> 2
excellent -> 3
```

For:

```text id="x5k9m3"
values = ["excellent", "poor", "good", "fair"]
```

the result is:

```text id="n1q6z8"
[3, 0, 2, 1]
```

---

# Worked Example: Unknown Value

Suppose:

```python id="r4w7c2"
values = ["small", "extra-large", "medium"]
order = ["small", "medium", "large"]
```

The mapping is:

```text id="b8j3p5"
small  -> 0
medium -> 1
large  -> 2
```

`extra-large` is unknown.

Therefore:

```text id="k2v9m6"
[0, -1, 1]
```

---

# Ordinal Encoding as a Function

The entire transformation can be expressed mathematically.

Given:

$$C=[c_0,c_1,\ldots,c_{n-1}]$$

define:

$$f(c_i)=i$$

For an input sequence:

$$V=[v_1,v_2,\ldots,v_m]$$

the output is:

$$E=[e_1,e_2,\ldots,e_m]$$

where:

$$e_j=
\begin{cases}
i & \text{if }v_j=c_i\\
-1 & \text{if }v_j\notin C
\end{cases}$$

The transformation is deterministic.

The same category always receives the same integer as long as the
ordering remains unchanged.

---

# Important Implementation Details

## The Order Is the Source of Truth

The function does not infer ordering from the input.

For:

```text id="w2m7q5"
values = ["large", "small", "medium"]
```

the order in which values appear does not matter.

The ordering is explicitly defined by:

```text id="n4x8c1"
order = ["small", "medium", "large"]
```

Therefore:

```text id="z6r3v9"
large  -> 2
small  -> 0
medium -> 1
```

---

## Do Not Sort Categories Automatically

A common mistake is to derive the mapping with:

```python id="f0y4s7"
sorted(set(values))
```

This can destroy the semantic ordering.

For example:

```text id="m5k2x8"
["high school", "bachelor", "master", "phd"]
```

should not be ordered alphabetically.

The explicit `order` list allows domain knowledge to determine the
ranking.

---

## Order Must Contain Unique Categories

The problem guarantees that `order` contains unique categories.

Therefore:

```text id="c7v1p4"
["small", "medium", "large"]
```

is valid.

A duplicate ordering such as:

```text id="q9n3z6"
["small", "medium", "small"]
```

would create an ambiguous mapping.

With a dictionary comprehension, the later occurrence would overwrite
the earlier one.

The problem avoids this ambiguity through its constraint.

---

## Unknown Categories and Model Semantics

Although `-1` is convenient for this problem, downstream models may
interpret it as a genuinely ordered numerical value.

For example:

```text
unknown -> -1
small   -> 0
medium  -> 1
large   -> 2
```

A linear model could treat unknown as being below `small`.

That may not be semantically correct.

A production pipeline should therefore decide whether unknown should be
represented using a separate category or another encoding strategy.

---

## Ordinal Encoding Does Not Create Information

Suppose:

```text
poor -> 0
fair -> 1
good -> 2
excellent -> 3
```

The encoding does not create the ranking.

The ranking already exists in the original categorical variable.

The encoding merely makes that known ordering available in numerical
form.

This is a general principle in feature engineering:

```text
Encoding should preserve useful structure,
not invent structure that does not exist.
```

---

> 💡 **Interview Tip**
>
> If asked why ordinal encoding is different from ordinary label
> encoding, focus on the source of the numerical order. In ordinal
> encoding, the integer values are deliberately assigned according to a
> meaningful domain ranking. Arbitrary integer labels for nominal
> categories can introduce a false ordering.

---

# Complexity Analysis

Let:

- $n$ = number of categories in `order`
- $m$ = number of values to encode

Building the dictionary requires visiting every category:

$$O(n)$$

Encoding each input value requires an average constant-time dictionary
lookup:

$$O(1)$$

for each of the $m$ values.

Therefore:

$$O(m)$$

is required for encoding.

The total expected time complexity is:

$$O(n+m)$$

---

## Space Complexity

The dictionary stores one mapping for each category:

$$O(n)$$

The output list contains one integer for every input value:

$$O(m)$$

Therefore, total space including the returned output is:

$$O(n+m)$$

The auxiliary mapping itself requires:

$$O(n)$$

space.

---

## Final Complexity

| Complexity | Value |
| ---------- | ----- |
| Time | **O(n + m)** |
| Auxiliary Space | **O(n)** |
| Output Space | **O(m)** |
| Total Space | **O(n + m)** |

where:

- $n$ = number of categories in `order`
- $m$ = number of values in `values`

The dictionary-based implementation is therefore both simple and
efficient for ordinal label encoding.
$$
