def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    # Your code here
    centroids = initial_centroids[:]
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        clusters = {c: [] for c in centroids}
        for point in points:
            nearest = min(
                centroids,
                key=lambda c: sum((x - y) ** 2 for x, y in zip(point, c))
            )
            clusters[nearest].append(point)
        # Update centroids
        new_centroids = []
        for centroid in centroids:
            cluster = clusters[centroid]
            if not cluster:  # empty cluster
                new_centroids.append(centroid)
                continue
            dim = len(cluster[0])
            new_centroid = tuple(
                round(sum(point[i] for point in cluster) / len(cluster), 4)
                for i in range(dim)
            )
            new_centroids.append(new_centroid)
        centroids = new_centroids
    return centroids

# TC: O(t*n*k*d)
# SC: O(n+k)