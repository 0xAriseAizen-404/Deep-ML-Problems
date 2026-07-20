import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    # if len(a[0]) != len(b):
    #     return -1
    # res = []
    # for vec in a:
    #     new_row = []
    #     for col in range(len(b[0])):
    #         new_row.append(sum(vec[k] * b[k][col] for k in range(len(b))))
    #     res.append(new_row)
    # return res

    try:
        return np.dot(np.asarray(a), np.asarray(b)).tolist()
    except ValueError:
        return -1