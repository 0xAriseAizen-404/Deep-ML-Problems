# K-Means Clustering (Medium, Machine Learning)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Example](#example)
- [Learn: K-Means Clustering](#learn-k-means-clustering)
- [Solutions](#solutions)
  - [Python Implementation](#python-implementation)
- [Code Explanation](#code-explanation)
- [Time & Space Complexity](#time--space-complexity)

---

## Problem Statement

[K-Means Clustering](https://www.deep-ml.com/problems/17)

Write a Python function to implement the **K-Means Clustering** algorithm. Given a set of points, initial centroids, and the maximum number of iterations, return the final centroids rounded to **4 decimal places**.

---

## Example

```python
Input:

points = [(1, 2), (1, 4), (1, 0),
          (10, 2), (10, 4), (10, 0)]

k = 2

initial_centroids = [(1, 1), (10, 1)]

max_iterations = 10

Output:

[(1, 2), (10, 2)]
```

**Reasoning**

Points are assigned to the nearest centroid, then each centroid is updated to the mean of its assigned points until the maximum iterations are reached.

---

## Learn: K-Means Clustering

K-Means is an **unsupervised learning** algorithm used to group similar data points into **k clusters**.

**Steps**

1. Initialize centroids.
2. Assign each point to the nearest centroid.
3. Compute the new centroid as the mean of its cluster.
4. Repeat until convergence or the maximum iterations.

Distance is usually measured using **Euclidean Distance**.

---

## Solutions

### Python Implementation

```python
def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    centroids = initial_centroids[:]

    for _ in range(max_iterations):

        clusters = {c: [] for c in centroids}

        for point in points:
            nearest = min(
                centroids,
                key=lambda c: sum((x - y) ** 2 for x, y in zip(point, c))
            )
            clusters[nearest].append(point)

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


# Example Usage
points = [(1, 2), (1, 4), (1, 0),
          (10, 2), (10, 4), (10, 0)]

print(k_means_clustering(
    points,
    2,
    [(1, 1), (10, 1)],
    10
))
```

---

## Code Explanation

- Initialize the centroids.
- Assign each point to its nearest centroid.
- Compute the mean of each cluster to obtain new centroids.
- Keep the previous centroid if a cluster becomes empty.
- Repeat for the specified number of iterations.

---

## Time & Space Complexity

| Complexity | Value |
|------------|-------|
| Time | **O(iterations × n × k × d)** |
| Space | **O(n + k)** |

Where:

- **n** = Number of points
- **k** = Number of clusters
- **d** = Number of dimensions