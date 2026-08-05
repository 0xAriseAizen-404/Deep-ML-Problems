import math
import cmath
def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[complex]:
    if len(matrix[0]) != 2:
        raise ValueError("Only 2x2 matrices are supported")
        # EigenValues => lamba**2 - trace(A)*lambda + det(A) = 0
        # This is a Quadratice Equation, we need to find the roots here
        # (-b += sqrt(b**2 - 4ac)) / 2a
        # a, b, c = 1, -trace, determinent
        # root1 = (-b + ((b**2) - (4*a*c))**0.5) / (2*a)
        # root2 = (-b - ((b**2) - (4*a*c))**0.5) / (2*a)
        
    trace = matrix[0][0] + matrix[1][1]
    trace = -trace # x coeff is -b in the equation
    determinant = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    disc = trace**2 - 4 * determinant
    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
    else:
        sqrt_disc = cmath.sqrt(disc)
    root1 = (-trace + sqrt_disc) / 2
    root2 = (-trace - sqrt_disc) / 2
    return [root1, root2]

# TC: O(1)
# SC: O(1)