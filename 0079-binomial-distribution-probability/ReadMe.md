# Binomial Distribution Probability (Medium, Probability)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: Binomial Distribution](#learn-binomial-distribution)
  - [What is the Binomial Distribution?](#what-is-the-binomial-distribution)
  - [Bernoulli Trials](#bernoulli-trials)
  - [Conditions for a Binomial Distribution](#conditions-for-a-binomial-distribution)
  - [Probability Mass Function](#probability-mass-function)
  - [Binomial Coefficient](#binomial-coefficient)
  - [Why the Formula Works](#why-the-formula-works)
  - [Example Calculation](#example-calculation)
  - [Mean and Variance](#mean-and-variance)
  - [Properties](#properties)
  - [Edge Cases](#edge-cases)
  - [Applications](#applications)

- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
  - [Python Implementation](#python-implementation)

- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [Binomial Distribution Probability](https://www.deep-ml.com/problems/79)

Implement a function that calculates the probability of achieving exactly $k$ successes in $n$ independent Bernoulli trials.

Each trial has the same probability $p$ of success.

The random variable $X$ represents the total number of successes.

Therefore:

\(X\sim Binomial(n,p)\)

The required probability is:

\(P(X=k)\)

The function should use the Binomial probability mass function to calculate this value.

## Example

### Input

```python id="xq8m4s"
n = 6
k = 2
p = 0.5
```

### Output

```text id="y9v3kp"
0.23438
```

### Reasoning

We want exactly $2$ successes in $6$ independent trials.

The number of possible arrangements containing exactly two successes is:

\(\binom{6}{2}=15\)

The probability of any particular arrangement containing two successes and four failures is:

\(0.5^2(1-0.5)^4\)

which is:

\(0.25\times0.0625=0.015625\)

There are $15$ such arrangements.

Therefore:

\(P(X=2)=15\times0.25\times0.0625\)

\(P(X=2)=0.234375\)

Rounded to five decimal places:

\(P(X=2)\approx0.23438\)

## Learn: Binomial Distribution

### What is the Binomial Distribution?

The Binomial distribution is a discrete probability distribution that models the number of successes obtained from a fixed number of independent trials.

Each trial has exactly two possible outcomes:

- Success
- Failure

If there are $n$ trials and each trial has probability $p$ of success, then:

\(X\sim Binomial(n,p)\)

The distribution answers questions such as:

> What is the probability of getting exactly $k$ successes?

Examples include:

- Exactly 7 defective products in 100 inspected products.
- Exactly 3 successful treatments in 10 trials.
- Exactly 20 conversions from 50 website visitors.
- Exactly 4 heads in 10 coin flips.

### Bernoulli Trials

The Binomial distribution is built from Bernoulli trials.

A Bernoulli random variable has only two possible outcomes.

For a single trial:

\(X\in\{0,1\}\)

where:

\(P(X=1)=p\)

and:

\(P(X=0)=1-p\)

A value of $1$ represents success.

A value of $0$ represents failure.

The Binomial distribution counts how many successes occur when multiple Bernoulli trials are performed.

### Conditions for a Binomial Distribution

A random variable follows a Binomial distribution when the following conditions hold.

#### Fixed Number of Trials

The number of trials must be fixed.

\(n=\text{constant}\)

For example, flipping a coin exactly 10 times satisfies this condition.

#### Two Possible Outcomes

Each trial must have two possible outcomes.

These are usually called success and failure.

The outcomes do not necessarily have to be literally "success" and "failure."

For example:

- Defective / not defective.
- Click / no click.
- Pass / fail.
- Conversion / no conversion.

#### Independence

The outcome of one trial must not affect another trial.

For independent trials:

\(P(X_i|X_j)=P(X_i)\)

for different trials.

For example, repeated fair coin flips are normally modeled as independent.

#### Constant Probability

Every trial must have the same probability of success.

\(P(success)=p\)

for every trial.

If the probability changes from trial to trial, the standard Binomial model no longer applies directly.

### Probability Mass Function

The Binomial probability mass function is:

\(P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\)

This formula gives the probability of exactly $k$ successes.

The terms have specific meanings.

\(\binom{n}{k}\)

counts how many arrangements contain exactly $k$ successes.

\(p^k\)

is the probability of obtaining $k$ successes.

\(1-p\)

is the probability of a failure.

\(n-k\)

is the number of failures.

Therefore:

\((1-p)^{n-k}\)

is the probability of obtaining the required number of failures.

### Binomial Coefficient

The Binomial coefficient is written as:

\(\binom{n}{k}\)

and pronounced:

> "n choose k"

It represents the number of ways to select $k$ positions from $n$ positions.

The formula is:

\(\binom{n}{k}=\frac{n!}{k!(n-k)!}\)

For example:

\(\binom{6}{2}=\frac{6!}{2!4!}\)

which gives:

\(\binom{6}{2}=15\)

Python provides this directly:

```python id="n2f8qa"
math.comb(n, k)
```

### Why the Binomial Coefficient is Necessary

Suppose we want exactly two successes in three trials.

The possible arrangements are:

```text id="e8m4rt"
S S F
S F S
F S S
```

There are three valid arrangements.

Therefore:

\(\binom{3}{2}=3\)

Each arrangement has probability:

\(p^2(1-p)\)

Therefore:

\(P(X=2)=3p^2(1-p)\)

The Binomial coefficient accounts for the fact that the successes can appear in different positions.

### Why the Formula Works

Consider exactly $k$ successes in $n$ trials.

Every valid sequence contains:

- $k$ successes.
- $n-k$ failures.

For one particular sequence, the probability is:

\(p^k(1-p)^{n-k}\)

However, there are multiple ways to arrange those successes.

The number of arrangements is:

\(\binom{n}{k}\)

Since the trials are independent, each arrangement has the same probability.

Therefore, multiply the probability of one arrangement by the number of valid arrangements:

\(P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\)

### Example Calculation

Consider:

\(n=5,\quad k=2,\quad p=0.4\)

#### Step 1: Binomial Coefficient

\(\binom{5}{2}=\frac{5!}{2!3!}=10\)

#### Step 2: Success Probability

There are two successes:

\(p^k=0.4^2=0.16\)

#### Step 3: Failure Probability

There are:

\(n-k=3\)

failures.

Therefore:

\((1-p)^{n-k}=0.6^3=0.216\)

#### Step 4: Combine the Terms

\(P(X=2)=10\times0.16\times0.216\)

Therefore:

\(P(X=2)=0.3456\)

### Interpreting the Probability

The value:

\(P(X=2)=0.3456\)

means there is a $34.56%$ probability of observing exactly two successes under the specified Binomial model.

It does not mean that two successes are guaranteed.

It also does not represent the probability of getting at least two successes.

The distinction between:

\(P(X=2)\)

and:

\(P(X\geq2)\)

is important.

The first is an exact probability.

The second requires summing multiple outcomes.

### Exact vs At Least vs At Most

The Binomial PMF directly calculates:

\(P(X=k)\)

For at least $k$ successes:

\(P(X\geq k)=\sum\_{i=k}^{n}P(X=i)\)

For at most $k$ successes:

\(P(X\leq k)=\sum\_{i=0}^{k}P(X=i)\)

For fewer than $k$ successes:

\(P(X<k)=\sum\_{i=0}^{k-1}P(X=i)\)

Therefore, this problem specifically asks for an exact probability.

### Complementary Probability

Sometimes it is easier to calculate an event using its complement.

For example:

\(P(X\geq k)=1-P(X<k)\)

This can reduce the number of terms that need to be calculated.

The same principle applies to many discrete probability calculations.

### Mean

For a Binomial random variable:

\(X\sim Binomial(n,p)\)

the expected value is:

\(E[X]=np\)

This represents the expected number of successes.

For example, with:

\(n=100,\quad p=0.2\)

the expected number of successes is:

\(E[X]=100(0.2)=20\)

The expected value does not imply that every experiment will produce exactly 20 successes.

### Variance

The variance of a Binomial random variable is:

\(Var(X)=np(1-p)\)

The standard deviation is:

\(\sigma=\sqrt{np(1-p)}\)

For:

\(n=100,\quad p=0.2\)

the variance is:

\(100(0.2)(0.8)=16\)

and the standard deviation is:

\(\sigma=4\)

### Properties

A valid Binomial probability satisfies:

\(P(X=k)\geq0\)

for every valid $k$.

All possible outcomes together must have probability one:

\(\sum\_{k=0}^{n}P(X=k)=1\)

The expected value is:

\(\mu=np\)

The variance is:

\(\sigma^2=np(1-p)\)

The standard deviation is:

\(\sigma=\sqrt{np(1-p)}\)

### Valid Range of k

The number of successes cannot be negative:

\(k\geq0\)

It also cannot exceed the number of trials:

\(k\leq n\)

Therefore:

\(0\leq k\leq n\)

For integer-valued trials, $k$ must be an integer.

If $k<0$ or $k>n$, the probability is:

\(P(X=k)=0\)

A robust implementation should validate these inputs.

### Valid Range of p

Since $p$ represents a probability:

\(0\leq p\leq1\)

If:

\(p<0\)

or:

\(p>1\)

the Binomial model is invalid.

The two extreme cases are especially useful.

If:

\(p=0\)

then success is impossible.

If:

\(p=1\)

then success occurs on every trial.

### Edge Case: p = 0

When:

\(p=0\)

the only possible outcome is zero successes.

Therefore:

\(P(X=0)=1\)

and:

\(P(X=k)=0\)

for $k>0$.

### Edge Case: p = 1

When:

\(p=1\)

every trial succeeds.

Therefore:

\(P(X=n)=1\)

and:

\(P(X=k)=0\)

for $k<n$.

### Edge Case: k = 0

Exactly zero successes means every trial is a failure.

Therefore:

\(P(X=0)=(1-p)^n\)

The binomial coefficient is:

\(\binom{n}{0}=1\)

and:

\(p^0=1\)

### Edge Case: k = n

Exactly $n$ successes means every trial succeeds.

Therefore:

\(P(X=n)=p^n\)

because:

\(\binom{n}{n}=1\)

and:

\((1-p)^0=1\)

### Relationship to Bernoulli Distribution

A Bernoulli distribution describes one trial.

A Binomial distribution describes the number of successes across $n$ independent Bernoulli trials.

Therefore, a Binomial random variable can be represented as a sum:

\(X=X_1+X_2+\cdots+X_n\)

where each:

\(X_i\sim Bernoulli(p)\)

and the variables are independent.

The expected value follows from linearity:

\(E[X]=\sum\_{i=1}^{n}E[X_i]=np\)

### Probability Generating Perspective

The Binomial distribution can also be derived from the expansion:

\((p+(1-p))^n=1\)

By the Binomial theorem:

\(\sum\_{k=0}^{n}\binom{n}{k}p^k(1-p)^{n-k}=1\)

Each term corresponds to the probability of exactly $k$ successes.

This directly explains why the probabilities sum to one.

### Shape of the Distribution

The shape depends strongly on $p$.

When:

\(p=0.5\)

the distribution is symmetric around:

\(n/2\)

When $p$ is small, the distribution tends to concentrate toward smaller values of $k$.

When $p$ is large, it concentrates toward larger values of $k$.

The mean is always:

\(np\)

but the most probable value, called the mode, need not equal the mean exactly.

### Relationship to Large Samples

For sufficiently large $n$, the Binomial distribution can often be approximated by a Normal distribution when the success probability is not too close to $0$ or $1$.

The corresponding Normal approximation has:

\(\mu=np\)

and:

\(\sigma^2=np(1-p)\)

A common practical condition is that both:

\(np\)

and:

\(n(1-p)\)

are sufficiently large.

For exact probabilities, however, the Binomial formula remains the direct model.

### Numerical Considerations

The factorial definition:

\(\binom{n}{k}=\frac{n!}{k!(n-k)!}\)

can involve extremely large intermediate values for large $n$.

Python's:

```python id="r6q4x1"
math.comb(n, k)
```

uses an integer-based implementation and avoids floating-point factorial calculations.

For very large probability problems, directly computing:

\(p^k(1-p)^{n-k}\)

can still cause numerical underflow.

Scientific libraries therefore often provide specialized Binomial PMF functions or logarithmic implementations.

### Applications

The Binomial distribution appears whenever we count successes among repeated independent binary trials.

Common applications include:

- Quality control.
- Defect detection.
- Medical trials.
- A/B testing.
- Survey response analysis.
- Conversion-rate analysis.
- Reliability testing.
- Classification outcomes.
- Manufacturing processes.
- Genetic inheritance models.

### Machine Learning Connection

Suppose a classifier predicts whether a user will click an advertisement.

Each user can be modeled as a Bernoulli trial:

\(X_i\sim Bernoulli(p)\)

If we observe $n$ independent users, the total number of clicks can be modeled as:

\(X\sim Binomial(n,p)\)

The probability of exactly $k$ clicks is:

\(P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\)

This connects discrete probability directly to binary classification and conversion modeling.

> 💡 **Important Note**
>
> The Binomial distribution models the **count of successes**, not the sequence of successes. The factor $\binom{n}{k}$ is necessary because the same number of successes can occur in many different orders.

## Solutions

### Custom Implementation

```python id="m8p3qd"
import math

def binomial_probability(n: int, k: int, p: float) -> float:
    def fact(n: int) -> int:
        if n == 0 or n == 1:
            return 1
        return n * fact(n - 1)

    def nck(n: int, k: int) -> float:
        return fact(n) / (fact(k) * fact(n - k))

    binomial_coefficient = math.comb(n, k) * math.pow(p, k) * math.pow(1 - p, n - k)

    return binomial_coefficient
```

### Python Implementation

The problem can be implemented more directly because Python already provides the Binomial coefficient through `math.comb`.

```python id="p4x7nv"
import math

def binomial_probability(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p)**(n - k)
```

The second version is preferable because it removes the unnecessary recursive factorial implementation.

For production code, input validation can also be added:

```python id="t1v6ks"
import math

def binomial_probability(n: int, k: int, p: float) -> float:
    if n < 0 or k < 0 or k > n:
        return 0.0

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    return math.comb(n, k) * p**k * (1 - p)**(n - k)
```

## Code Explanation

### 1. Compute the Number of Arrangements

```python id="a7m2xc"
math.comb(n, k)
```

This calculates:

\(\binom{n}{k}\)

It represents the number of different arrangements containing exactly $k$ successes.

For example:

\(\binom{6}{2}=15\)

### 2. Compute the Probability of the Successes

```python id="b8q4zt"
p**k
```

There are $k$ successes.

Each success has probability $p$.

Because the trials are independent:

\(P(\text{k successes})=p^k\)

### 3. Compute the Probability of the Failures

```python id="c5n9wy"
(1 - p)**(n - k)
```

There are:

\(n-k\)

failures.

The probability of one failure is:

\(1-p\)

Therefore:

\(P(\text{required failures})=(1-p)^{n-k}\)

### 4. Multiply the Terms

```python id="d6r1vp"
math.comb(n, k) * p**k * (1 - p)**(n - k)
```

This directly implements:

\(P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\)

The three components have distinct roles:

\(\text{number of arrangements}\times\text{probability of one arrangement}\)

### 5. Return the Probability

```python id="e2k8hs"
return math.comb(n, k) * p**k * (1 - p)**(n - k)
```

The resulting floating-point value is the probability of exactly $k$ successes.

For the problem example:

```python id="f9v3ma"
binomial_probability(6, 2, 0.5)
```

returns approximately:

```text id="g1q6zr"
0.234375
```

which rounds to:

```text id="h5t8kc"
0.23438
```

### Why the Recursive Factorial Helper Is Unnecessary

The submitted solution contains:

```python id="j4m7ps"
def fact(n: int) -> int:
    if n == 0 or n == 1:
        return 1
    return n * fact(n - 1)
```

and:

```python id="k3r9xd"
def nck(n: int, k: int) -> float:
    return fact(n) / (fact(k) * fact(n - k))
```

These functions correctly express the mathematical definition of the factorial and Binomial coefficient.

However, they are not actually used in the final calculation because the solution uses:

```python id="l8w2qa"
math.comb(n, k)
```

Therefore, the recursive helpers can be removed.

### Why math.comb Is Better

The expression:

```python id="n5x7vb"
math.comb(n, k)
```

is:

- Simpler.
- More readable.
- Less error-prone.
- Implemented efficiently by Python.
- Avoids explicit recursive factorial computation.

The key algorithmic idea is still the same:

\(\binom{n}{k}p^k(1-p)^{n-k}\)

### Complexity of Factorial-Based Computation

A naive recursive factorial computes:

\(n!\)

through $n$ recursive multiplications.

Therefore, its arithmetic operation count is:

\(O(n)\)

The recursive depth is also:

\(O(n)\)

For large $n$, this is undesirable.

Furthermore, the Binomial coefficient computes three factorials.

### Complexity of math.comb

Python's `math.comb` is designed specifically for computing exact Binomial coefficients and is substantially more appropriate than manually calculating factorials.

The exact implementation complexity depends on the integer arithmetic involved.

For ordinary problem-sized inputs, it is effectively the correct library primitive to use.

### Input Validation

The mathematically valid parameter ranges are:

\(n\geq0\)

\(0\leq k\leq n\)

\(0\leq p\leq1\)

If these constraints are guaranteed by the problem, explicit validation is unnecessary.

If the function is intended for general use, validation prevents invalid factorial and probability calculations.

### Important Difference Between Exact and Cumulative Probability

The function calculates:

\(P(X=k)\)

It does not calculate:

\(P(X\leq k)\)

or:

\(P(X\geq k)\)

For example, if we want at least two successes:

\(P(X\geq2)=P(X=2)+P(X=3)+\cdots+P(X=n)\)

A different calculation is therefore required.

## Time & Space Complexity

Let $n$ be the number of trials.

### Submitted Implementation

The submitted code defines a recursive factorial function.

Computing a factorial requires:

\(O(n)\)

recursive operations.

The `nck` helper would therefore require factorial calculations for $n$, $k$, and $n-k$.

However, `nck` is not actually called in the returned expression.

The final expression uses:

```python id="q4s8nc"
math.comb(n, k)
```

Therefore, the actual runtime is determined by `math.comb` and the exponentiation operations.

### Direct Formula

The conceptual calculation contains:

- One Binomial coefficient.
- One power operation for successes.
- One power operation for failures.
- A constant number of multiplications.

Thus, for typical problem inputs, the calculation is effectively constant in the number of trials apart from the integer arithmetic required by the Binomial coefficient.

### Space Complexity

The direct implementation uses only a constant number of scalar values.

Therefore, auxiliary space is:

\(O(1)\)

The recursive factorial helper, if used, would require:

\(O(n)\)

call-stack space.

### Complexity Table

| Operation                             | Complexity                       |
| ------------------------------------- | -------------------------------- |
| Direct probability formula            | **O(1)** scalar-level operations |
| Recursive factorial                   | **O(n)** operations              |
| Recursive factorial stack             | **O(n)**                         |
| Direct implementation auxiliary space | **O(1)**                         |

### Final Takeaway

The Binomial distribution answers:

> What is the probability of obtaining exactly $k$ successes in $n$ independent trials when every trial has success probability $p$?

The central formula is:

\(P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\)

The three components represent:

\(\binom{n}{k}=\text{number of valid arrangements}\)

\(p^k=\text{probability of the successes}\)

\((1-p)^{n-k}=\text{probability of the failures}\)

The submitted solution correctly implements this formula.

The main cleanup is removing the unused recursive `fact` and `nck` helpers and directly using Python's `math.comb`.
