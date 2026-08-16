import numpy as np

def cross_product(a, b):
    # return [a[1]*b[2] - a[2]*b[1], 
    # a[2]*b[0] - a[0]*b[2],
    # a[0]*b[1] - a[1]*b[0]]

    dot = np.dot(a, b)
    theta = np.arccos(dot / (np.linalg.norm(a) * np.linalg.norm(b))) # Dot is a Scalar

    cross = np.cross(a, b)
    theta = np.arcsin(np.linalg.norm(cross) / (np.linalg.norm(a) * np.linalg.norm(b))) # Cross is a Vector

    cross = np.cross(a, b)
    cross = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

    area = np.linalg.norm(a) * np.linalg.norm(b) * np.sin(theta)
    area = np.linalg.norm(cross)
    area = np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)

    return np.cross(a, b)

# TC: O(1)
# SC: O(1)


    
    # Cross Product of two vectors, is a Area of the paralellogram formed by those two vectors in 2D or 3D.
    # A X B = (-1)^(z) * Area of the Paralellogram
    # -1^z is basically a sign representing the positive or negative of the area
    # z = 1, if the Vector B is on right side (or Clockwise) of Vector A = Negative
    # z = 2, if the Vector B is on left side (or Anti-Clockwise) of Vector A = Positive
    # positive if BBB is anticlockwise from AAA
    # negative if BBB is clockwise from AAA
    # zero if they are collinear
    # 
    # how to find that area of the paralellogram,
    # its nothing but the Determinant formed from those two vectors in a space, right.
    # A = [a1 a2]
    # B = [b1 b2]
    # A X B = -1^z * (a1b2 - a2b1)
    # 
    # => Why Determinant Gives Area
    # The absolute value of a determinant measures how much area is scaled by the transformation.
    # AreaÃÂ ofÃÂ parallelogram = Ã¢ÂÂ£det(M)Ã¢ÂÂ£Ã¢ÂÂ
    # 
    # => In 3D Space
    # there is another information in terms of Cross product.
    # here in 3D space, two vectors A and B, its cross product gives Area of the Paralellogram, i.e, their Determinant
    # but, that Area of the Paralellogram or Determinant of those two vectors, will also nothing but the Magnitude of the Vector which is perpendicular to those both vectors, and its Direction is given by Right-Hand-Thumb-Rule, Pointing Vector A in Point Finger direction and Pointing Vector B in Middle Finger direction and the direction of Right Hand Thumb finger gives the direction of that Vector A X B.
    # So, A X B = A Vector perpendicular to both vectors & having right-hand-thumb-rule direction
    # |A X B| = Area of the Paralellogram or Determinant of two vectors A & B
    # Vector representation of the A X B can be get with,
    # [i a1 b1]
    # [j a2 b2]
    # [k a3 b3]
    # Determinant along column 0,
    # A X B = i(a2b3-a3b2) - j(a3b1-a1b3) + k(a1b2-a2b1) = Vector Representation of A X B
    # |A X B| = sqrt((a2b3-a3b2)**2 + (a3b1-a1b3)**2 + (a1b2-a2b1)**2) = Area of the Paralellogram A X B = Magnitude of A X B
    # |A X B| = |A||B|Sin(theta)
    # 
    # In 2D:
    # A ÃÂ B = a1b2 - a2b1
    # which is exactly the determinant:
    # | a1  b1 |
    # | a2  b2 |
    # Hence, Area = |Determinant|
    # In 3D:
    # A and B together form a 3ÃÂ2 matrix:
    # | a1 b1 |
    # | a2 b2 |
    # | a3 b3 |
    # A determinant is defined only for square matrices.
    # Since a 3ÃÂ2 matrix is NOT square,
    # "Determinant of A and B" is NOT defined.
    # Therefore, |A ÃÂ B| Ã¢ÂÂ  Determinant(A,B)
    # The cross product vector is built using several 2ÃÂ2 determinants, but its magnitude itself is not a determinant.