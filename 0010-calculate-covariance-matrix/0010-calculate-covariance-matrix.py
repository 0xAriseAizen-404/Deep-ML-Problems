import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    # Your code here
    # mean_features = [sum(vec)/len(vec) for vec in vectors]

    # def cov(x, y):
    #     mean_x, mean_y = sum(x)/len(x), sum(y)/len(y)
    #     return sum([(x[k]-mean_x)*(y[k]-mean_y) for k in range(len(x))]) / (len(x)-1)
    
    # n = len(vectors)
    # res = [[1.0 for _ in range(n)] for _ in range(n)]
    # for i in range(n):
    #     for j in range(i, n):
    #         val = cov(vectors[i], vectors[j])
    #         res[i][j] = val
    #         res[j][i] = val
    # return res

    return np.cov(vectors).tolist()

# TC: O(m^2 * n)
# SC: O(m^2)