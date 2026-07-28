import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # N = X.shape[0]
    # # def generate(N, k):
    #     # return np.random.choice(N, size=k, replace=(k > N)).tolist()
    # res = []
    # for _ in range(n_subsets):
    #     if replacements:
    #         # indexes = generate(sz, N)
    #         indexes = np.random.choice(N, size=N, replace=True)
    #     else:
    #         # indexes = generate(sz, N//2)
    #         indexes = np.random.choice(N, size=N//2, replace=False)
    #     res.append((X[indexes].tolist(), y[indexes].tolist()))
    # return res

    n, m = X.shape
    subset_size = n if replacements else n // 2
    idx = np.array([np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)])
    # convert all ndarrays to lists
    return [(X[idx[i]].tolist(), y[idx[i]].tolist()) for i in range(n_subsets)]