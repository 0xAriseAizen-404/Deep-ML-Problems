# K-Means Clustering (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: K-Means Clustering](#learn-k-means-clustering)
- [Solutions](#solutions)
  - [Custom Implementation](#custom-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

## Problem Statement

### [K-Means Clustering](https://www.deep-ml.com/problems/17)

Write a Python function that implements the **K-Means Clustering** algorithm.

The function receives:

- A collection of data points.
- The number of clusters **k**.
- Initial centroid locations.
- Maximum number of iterations.

The algorithm repeatedly assigns every point to its nearest centroid and updates each centroid as the mean of its assigned points. If a cluster becomes empty, its centroid should remain unchanged.

Return the final centroid coordinates rounded to **4 decimal places**.

---

## Example

### Input

```python
points = [
    (1, 2),
    (1, 4),
    (1, 0),
    (10, 2),
    (10, 4),
    (10, 0)
]

k = 2

initial_centroids = [
    (1, 1),
    (10, 1)
]

max_iterations = 10
```

### Output

```python
[
    (1, 2),
    (10, 2)
]
```

### Reasoning

Initially, every point is assigned to its closest centroid.

Cluster 1

```text
(1,2)
(1,4)
(1,0)
```

Mean

$$
\left(
\frac{1+1+1}{3},
\frac{2+4+0}{3}
\right) =
(1,2)
$$

Cluster 2

```text
(10,2)
(10,4)
(10,0)
```

Mean

$$
\left(
\frac{10+10+10}{3},
\frac{2+4+0}{3}
\right) =
(10,2)
$$

These become the updated centroids.

---

## Learn: K-Means Clustering

### What is K-Means Clustering?

K-Means is an **unsupervised Machine Learning** algorithm that partitions a dataset into **k clusters**.

Each cluster is represented by its **centroid**, which is the average location of all points assigned to that cluster.

The objective is to place similar data points into the same cluster while keeping different clusters as separate as possible.

Unlike supervised learning, K-Means does **not** require labeled data.

---

### Intuition

Imagine dropping **k pins** randomly on a map.

Then repeatedly perform two operations:

1. Assign every location to its nearest pin.
2. Move each pin to the average location of its assigned points.

Eventually the pins stabilize and represent the centers of natural groups in the data.

---

### Euclidean Distance

To determine the closest centroid, K-Means usually uses **Euclidean Distance**.

For two points

$$
x=(x_1,x_2,\ldots,x_n)
$$

and

$$
y=(y_1,y_2,\ldots,y_n)
$$

their distance is

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

Since the square root is monotonic, implementations often compare the squared distance

$$
d^2(x,y)=\sum_{i=1}^{n}(x_i-y_i)^2
$$

to avoid unnecessary computation.

---

### Centroid Update

After assigning points to clusters, the centroid is updated by computing the mean of every coordinate.

For a cluster containing $m$ points,

$$
c=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

For two-dimensional data,

$$
c=
\left(
\frac{\sum x_i}{m},
\frac{\sum y_i}{m}
\right)
$$

The centroid does not have to be one of the original data points.

---

### K-Means Objective Function

The algorithm minimizes the **Within-Cluster Sum of Squares (WCSS)**.

$$
J=\sum_{i=1}^{k}\sum_{x\in C_i}\|x-\mu_i\|^2
$$

where

- $C_i$ is the $i^{th}$ cluster.
- $\mu_i$ is its centroid.

The goal is to make points inside each cluster as close as possible to their centroid.

---

### Algorithm

1. Choose **k** initial centroids.
2. Assign every point to its nearest centroid.
3. Compute new centroids using the cluster means.
4. Repeat until:
   - Maximum iterations are reached, or
   - Centroids stop changing significantly.

---

### Characteristics / Key Points

- Unsupervised learning algorithm.
- Distance-based clustering method.
- Uses centroid (mean) to represent a cluster.
- Works with numerical features.
- Sensitive to the choice of initial centroids.
- Different initializations may produce different results.
- Empty clusters are possible and must be handled appropriately.
- Works best for compact, roughly spherical clusters.

---

### Choosing the Value of K

Selecting the correct value of **k** is important.

Common techniques include:

- Elbow Method
- Silhouette Score
- Domain knowledge

A value of **k** that is too small merges unrelated groups, while a value that is too large splits natural clusters.

---

### Applications

K-Means is widely used in Machine Learning and Data Science.

Common applications include:

- Customer segmentation
- Image compression
- Color quantization
- Document clustering
- Recommendation systems
- Market basket analysis
- Anomaly detection preprocessing
- Geographic clustering

---

> 💡 **Important Note**
>
> K-Means is highly sensitive to **feature scaling** because it relies on Euclidean distance. Always standardize or normalize features before clustering. In practice, **K-Means++** initialization is preferred over random initialization because it usually converges faster and produces better clusters.

---

## Solutions

### Custom Implementation

```python
def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    centroids = initial_centroids[:]

    for _ in range(max_iterations):

        # Assign points to nearest centroid
        clusters = {c: [] for c in centroids}

        for point in points:
            nearest = min(
                centroids,
                key=lambda c: sum(
                    (x - y) ** 2
                    for x, y in zip(point, c)
                )
            )
            clusters[nearest].append(point)

        # Update centroids
        new_centroids = []

        for centroid in centroids:
            cluster = clusters[centroid]

            if not cluster:
                new_centroids.append(centroid)
                continue

            dim = len(cluster[0])

            new_centroid = tuple(
                round(
                    sum(point[i] for point in cluster) / len(cluster),
                    4
                )
                for i in range(dim)
            )

            new_centroids.append(new_centroid)

        centroids = new_centroids

    return centroids
```

---

## Code Explanation

### 1. Initialize the Centroids

```python
centroids = initial_centroids[:]
```

The provided initial centroids are copied so the original list remains unchanged.

---

### 2. Iterate for the Given Number of Steps

```python
for _ in range(max_iterations):
```

Each iteration performs one complete **Assignment Step** followed by one **Update Step**.

---

### 3. Assign Every Point to the Nearest Centroid

```python
nearest = min(
    centroids,
    key=lambda c: ...
)
```

For each point, compute the squared Euclidean distance to every centroid.

The centroid with the smallest distance is selected.

This corresponds to

$$
\arg\min_j \|x-\mu_j\|^2
$$

---

### 4. Build the Clusters

```python
clusters[nearest].append(point)
```

Every point is stored inside the cluster of its nearest centroid.

After this step, each cluster contains all assigned samples.

---

### 5. Compute New Centroids

For every cluster,

```python
sum(point[i] for point in cluster) / len(cluster)
```

calculates the average value of each coordinate.

This implements

$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

The resulting coordinates are rounded to four decimal places.

---

### 6. Handle Empty Clusters

```python
if not cluster:
    new_centroids.append(centroid)
```

If no points are assigned to a centroid, its previous position is retained.

This avoids division by zero and allows the algorithm to continue.

---

### 7. Update the Centroids

```python
centroids = new_centroids
```

The newly computed centroids replace the previous ones before the next iteration begins.

After the specified number of iterations, the final centroids are returned.

---

## Time & Space Complexity

Let

- $n$ = number of data points.
- $k$ = number of clusters.
- $d$ = number of dimensions.
- $t$ = maximum number of iterations.

Each iteration computes the distance from every point to every centroid and updates each centroid once.

| Complexity | Value                |
| ---------- | -------------------- |
| Time       | **O(t × n × k × d)** |
| Space      | **O(n + k)**         |

The additional space is used to store the clusters and centroid coordinates.
