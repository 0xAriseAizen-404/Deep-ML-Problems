# [Poisson Distribution Probability Calculator](https://www.deep-ml.com/problems/81) (Easy, Probability)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Understanding Poisson Distribution](#understanding-poisson-distribution)
  - [Probability Mass Function](#probability-mass-function)
  - [Understanding Lambda and K](#understanding-lambda-and-k)
  - [Key Properties](#key-properties)
  - [When to Use Poisson Distribution](#when-to-use-poisson-distribution)
  - [Example Calculation](#example-calculation)
- [Solution](#solution)
  - [Approach](#approach)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Complexity Analysis](#complexity-analysis)
- [Key Takeaways](#key-takeaways)

## Problem Statement

Implement a Python function that calculates the probability of observing exactly $k$ events in a fixed interval using the Poisson distribution.

The function takes:

- $k$: Number of events.
- $\lambda$: Mean number of events in the interval.

The probability should be rounded to 5 decimal places.

The Poisson probability mass function is:

$$P(K=k;\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}$$

## Example

### Input

```python
k = 3
lam = 5
```

### Output

```text
0.14037
```

### Reasoning

Using:

\(P(K=3;\lambda=5)=\frac{5^3e^{-5}}{3!}\)

\(=\frac{125e^{-5}}{6}\)

\(\approx0.14037\)

Therefore, the probability of observing exactly 3 events when the expected number of events is 5 is approximately `0.14037`.

## Understanding Poisson Distribution

The Poisson distribution is a discrete probability distribution used to model the number of times an event occurs within a fixed interval of time, space, or another defined unit.

It is useful when:

- Events occur independently.
- The average rate of occurrence is approximately constant.
- We are counting how many times an event occurs.
- The number of events in one interval does not directly determine the number in another interval.

For example, suppose a shop observes an average of 5 customers per hour.

Here:

\(\lambda=5\)

We can use the Poisson distribution to estimate the probability of getting exactly 3 customers, exactly 7 customers, or any other non-negative number of customers in the next hour.

### Probability Mass Function

The Poisson probability mass function is:

\(P(K=k;\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\)

where:

- $P(K=k)$ is the probability of exactly $k$ events.
- $k$ is the number of observed events.
- $\lambda$ is the average number of events per interval.
- $e$ is Euler's number, approximately $2.71828$.
- $k!$ is the factorial of $k$.

The formula contains three important components:

\(\lambda^k\)

controls the effect of the expected event rate,

\(e^{-\lambda}\)

provides the exponential decay term, and

\(k!\)

normalizes the probability for the number of events.

### Understanding Lambda and K

The distinction between $k$ and $\lambda$ is important.

#### Lambda

$\lambda$ represents the expected or average number of events in the interval.

For example:

```text
Average customers per hour = 5
```

gives:

\(\lambda=5\)

#### K

$k$ represents the exact number of events whose probability we want to calculate.

For example:

```text
Probability of exactly 3 customers
```

gives:

\(k=3\)

Therefore, $\lambda$ describes the expected rate, while $k$ describes the particular outcome being evaluated.

### Key Properties

#### Mean

For a Poisson random variable:

\(E[K]=\lambda\)

The expected number of events equals the rate parameter.

#### Variance

The variance is also:

\(Var(K)=\lambda\)

This is a distinctive property of the Poisson distribution.

Consequently, the standard deviation is:

\(\sigma=\sqrt{\lambda}\)

#### Discrete Outcomes

The Poisson distribution describes counts, so:

\(k\in\{0,1,2,3,\ldots\}\)

A fractional number of events such as $k=2.5$ does not make sense in this model.

### When to Use Poisson Distribution

Poisson distribution is commonly used for counting events in a fixed interval.

Examples include:

- Number of customers arriving at a store per hour.
- Number of calls received by a call center per minute.
- Number of emails received per hour.
- Number of accidents at an intersection per month.
- Number of mutations occurring in a DNA segment.
- Number of network requests arriving at a server per second.

The key idea is that we are interested in **how many times an event occurs**, rather than the exact timing of each event.

### Example Calculation

Suppose a store receives an average of 5 customers per hour:

\(\lambda=5\)

We want the probability of exactly 3 customers:

\(k=3\)

Substitute into the Poisson formula:

\(P(K=3;5)=\frac{5^3e^{-5}}{3!}\)

Calculate the individual terms:

\(5^3=125\)

\(3!=6\)

Therefore:

\(P(K=3;5)=\frac{125e^{-5}}{6}\)

Using:

\(e^{-5}\approx0.0067379\)

we obtain:

\(P(K=3;5)\approx0.14037\)

So there is approximately a **14.037%** probability of observing exactly 3 events.

## Solution

### Approach

The implementation directly follows the mathematical formula:

\(P(K=k;\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\)

The algorithm is:

1. Compute $\lambda^k$.
2. Compute $e^{-\lambda}$.
3. Compute $k!$.
4. Divide the numerator by the factorial.
5. Round the result to 5 decimal places.

Python's `math` module provides the required mathematical operations.

### Python Implementation

```python
import math

def poisson_probability(k, lam):
    return round(((lam ** k) * math.exp(-lam)) / math.factorial(k), 5)
```

## Code Explanation

### Compute $\lambda^k$

```python
lam ** k
```

This calculates the expected-rate term:

\(\lambda^k\)

For `lam = 5` and `k = 3`:

```text
125
```

### Compute $e^{-\lambda}$

```python
math.exp(-lam)
```

implements:

\(e^{-\lambda}\)

For `lam = 5`:

```text
e^-5 ≈ 0.0067379
```

### Compute $k!$

```python
math.factorial(k)
```

calculates:

\(k!\)

For `k = 3`:

\(3!=6\)

### Combine the Terms

```python
((lam ** k) * math.exp(-lam)) / math.factorial(k)
```

directly implements:

\(\frac{\lambda^k e^{-\lambda}}{k!}\)

### Round the Result

```python
round(..., 5)
```

returns the probability to five decimal places as required by the problem.

## Complexity Analysis

For a single probability calculation, the number of arithmetic operations is effectively constant with respect to the number of observations.

### Time Complexity

\(\boxed{O(1)}\)

The implementation performs a fixed number of mathematical operations.

### Space Complexity

\(\boxed{O(1)}\)

Only a constant amount of additional memory is required.

## Key Takeaways

- Poisson distribution models the number of events occurring in a fixed interval.
- $\lambda$ represents the average number of events.
- $k$ represents the exact number of events being evaluated.
- The probability formula is:

\(P(K=k;\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\)

- For a Poisson distribution:

\(E[K]=Var(K)=\lambda\)

- The outcomes are non-negative integers.
- `math.exp()` computes the exponential term.
- `math.factorial()` computes $k!$.
- The implementation directly translates the mathematical definition into Python.
