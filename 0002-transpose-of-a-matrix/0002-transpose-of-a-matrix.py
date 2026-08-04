def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    # max_len = max((len(row) for row in a), default=0)
    # res = [[] for _ in range(max_len)]
    # for i in range(len(a)):
    #     for j in range(len(a[i])):
    #         res[j].append(a[i][j])
    # return res

    m = max(len(row) for row in a)
    return [[row[i] for row in a if i < len(row)] for i in range(m)]

# TC: O(r*c)
# SC: O(r*c)