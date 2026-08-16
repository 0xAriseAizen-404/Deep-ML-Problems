def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float] | int:
    # Return the element-wise sum of vectors 'a' and 'b'.
    # If vectors have different lengths, return -1.
    if len(a) != len(b):
        return -1
    
    # res = list(a)
    # for i in range(len(b)):
        # res[i] += b[i]
    # return res
    
    return [x + y for x, y in zip(a, b)]

# TC: O(n)
# SC: O(n)