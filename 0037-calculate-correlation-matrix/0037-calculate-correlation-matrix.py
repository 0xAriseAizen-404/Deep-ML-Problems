import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Optimized Version
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    return np.asarray((X.T @ Y) / (X.shape[0] - 1))

    # # Simplified Version
    # X = np.asarray(X).T
    # if Y is None:
    #     Y = X
    # else:
    #     Y = np.asarray(Y).T
	# corr_matrix = []
    # for f1 in X:
    #     row = []
    #     for f2 in Y:
    #         cov = np.sum((f1 - f1.mean()) * (f2 - f2.mean())) / (len(f1) - 1)
    #         corr = cov / (f1.std(ddof=1) * f2.std(ddof=1))
    #         row.append(corr)
    #     corr_matrix.append(row)
    # return np.asarray(corr_matrix)

    # # My Version
    # corr_matrix = []
    # if Y is not None:
    #     X = np.asarray(X).T
    #     Y = np.asarray(Y).T
    #     for i in range(len(X)):
    #         row = []
    #         for j in range(len(Y)):
    #             f1, f2 = X[i], Y[j]
    #             cov = np.sum((f1 - f1.mean()) * (f2 - f2.mean())) / (len(f1) - 1)
    #             row.append(cov / (f1.std(ddof=1) * f2.std(ddof=1)))
    #         corr_matrix.append(row)
    # else:
    #     X = np.asarray(X).T
    #     for i in range(len(X)):
    #         row = []
    #         for j in range(len(X)):
    #             f1, f2 = X[i], X[j]
    #             cov = np.sum((f1 - f1.mean()) * (f2 - f2.mean())) / (len(f1) - 1)
    #             row.append(cov / (f1.std(ddof=1) * f2.std(ddof=1)))
    #         corr_matrix.append(row)
    # return np.asarray(corr_matrix)

# TC: O(nd^2)
# SC: O(nd + d^2)