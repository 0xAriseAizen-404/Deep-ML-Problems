def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]):
    if not a:
        return []

    if len(a[0]) != len(b):
        return -1

    res = []
    for row in a:
        total = 0
        for x, y in zip(row, b):
            total += x * y
        res.append(total)

    return res

# TC: O(m*n)
# SC: O(m)