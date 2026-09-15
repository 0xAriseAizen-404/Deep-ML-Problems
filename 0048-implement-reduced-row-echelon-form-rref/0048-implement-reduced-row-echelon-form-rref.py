import numpy as np
def rref(mat):
    matrix = mat.copy().astype(float)
    rows, cols = matrix.shape
    pivot = 0
    col = 0
    while pivot < rows and col < cols:
        if matrix[pivot][col] == 0:
            row_ind = -1
            for temp in range(pivot + 1, rows):
                if matrix[temp][col] != 0:
                    row_ind = temp
                    break
            if row_ind == -1:
                col += 1
                continue
            matrix[[pivot, row_ind]] = matrix[[row_ind, pivot]]
        matrix[pivot] = matrix[pivot] / matrix[pivot][col]
        for row in range(rows):
            if row != pivot:
                matrix[row] = matrix[row] - (matrix[row][col] * matrix[pivot])
        pivot += 1
        col += 1
    return matrix

# TC: O(mn min(m,n))
# SC: O(mn)

# RREF, Reduced Row Echelon Form
# lets see, first we have linear combinations like
# x+2y-z = -4
# 2x+3y-z = -11
# -2x-3z = 22
# Our Aim, is to solve these linera combinations
# for that matter, we take the coefficients of these into a matrix form and then we perform some Gauss-Jordan Elimination, we try to make that matrix into a Identity matrix.
# Gauss-Jordan Elimination ->
# for each column
# make the Pivot = 1 & make the above elements into zeroes & make the below elements into zeroes
# pivot is nothing but [col][col] index value
# 1 0 0 a
# 0 1 0 b
# 0 0 1 c
# Ax = Z
# [1 2 -1] [x] = [-4]
# [2 3 -1] [y] = [-11]
# [-2 0 -3] [z] = [22]
# Now, the matrix we take here is
# [1 2 -1 -4]
# [2 3 -1 -11]
# [-2 0 -3 22]
# now, we try to make the columns as pivotable as possible, based on the Guass-Jordan Elimination method transformations
# for col 0 -> pivot = [0][0]
# for col 1 -> pivot = [1][1]
# for col 2 -> pivot = [2][2]
# all the values need to become '1'
# and then the above and below of values of these pivots needs to become zeroes
# 
# For Pivot Rows, PivotRow = PivotRow / PivotEle, this makes it '1'
# For other rows, Row[i] = Row[i] - (Row[i][col] * PivotRow), this makes them '0'
# 
# First we take Column 0, and transform it into our target column
# R1 is pivot here
# For the Pivot rows to transform, simply divide that row with the pivot element
# R[0] = R[0] / R[0][0]
# R[1] = R[1] - (R[1][0] * R[0])
# R[2] = R[2] - (R[2][0] * R[0])
# Now, Column 1
# R[1] = R[1] / R[1][1]
# R[0] = R[0] - (R[0][1] * R[1])
# R[2] = R[2] - (R[2][1] * R[1])
# Now, Column 2
# R[2] = R[2] / R[2][2]
# R[0] = R[0] - (R[0][2] * R[2])
# R[1] = R[1] - (R[1][2] * R[2])
# 
# after doing the above transformations
# if any row has complete zeroes, then we move them to the end of the matrix
# 
# then we will get a matrix, that might be Identity or not
# if it is identity,
# [1 0 0 -8]
# [0 1 0 1]
# [0 0 1 -2]
# then the solution will be, one solution
# X = -8, y = 1, z = -2
# 
# if it is like having a zeroes row
# [1 0 4 -8]
# [0 1 3 1]
# [0 0 0 0]
# then the solution will be, inifite solutions
# 0 = 0 => z = t
# x = -8 - 4t
# y = 1 - 3t
# 
# if it is like having impossible row (0 = non-zero)
# [1 0 4 -8]
# [0 1 3 1]
# [0 0 0 5]
# 0 = 5, impossible
# so No Solution
# 
# RREF = Identity
    # => Unique solution
# RREF has zero row
    # => Maybe infinite solutions
# RREF has impossible row (0=non-zero)
    # => No solution
    # 
# Now, if incase, while we doing the transformations
# lets say, column 0's transformations are done
# then the matrix is looking like
# [1 2 4 -8]
# [0 0 3 1]
# [0 1 4 6]
# now, while doing transformations for column 2, the pivot element, mat[1][1] has 0
# we cant just do like, R[1] = R[1] / R[1][1], it not possible
# so we need to swap that row, with the below rows
# if all the below values in that column are zeroes, then we need to skip that column and move to next column at that specific pivot
# 1. If A[row][col] != 0
      # use it as pivot.
# 2. Else
      # search rows below.
# 3. If a non-zero value is found
      # swap rows.
# 4. If none found
      # this column has no pivot.
      # move to next column.
# 5. Continue until done.
# 
# Important Items
# using RREF, we can find the solution of Linear Combinations
# Rank(A) = Number of columns with Pivot leading 1's
# 
# ```
# Det(A) != 0
# Rank(A) = N
# RREF(A) = I
# A is Invertible, A.A^(-1) = I
# Columns are linearly independent
# ```
# If one is true, all are true.
# If one is false, all are false.