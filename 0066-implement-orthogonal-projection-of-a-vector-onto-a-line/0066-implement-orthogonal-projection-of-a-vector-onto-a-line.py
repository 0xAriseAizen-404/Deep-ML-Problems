import numpy as np
def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """

	# Go and watch some visualization on how Dot product work
	# Now, take two vectors A and B, attach them at their tail okay
	# and now, take a torch and place it at the tip of the Vector A, and try to place the shadow on the Vector B.
	# The point where the tip shadow is placed is the point where the projection happened
	# now, here the projection made a perpendicular to the Vector B, meaning a right angled triangle is formed with the angle of Theta at the attachment of the Vectors A and B.
	# The base of the triangle or the width of the triangle is Length of the Projection
	# now, Dot Product = Length of the Projection * Magnitude of the Vector on which the Projection happened (here its B)
	# lets consider to find the length of the projection,
	# as from Trigonometry fundamentals,
	# cos(theta) = adjacent / hypotenuse
	# cos(theta) = ProjectionLength / Magnitude of vector A
	# ProjectionLength = cos(theta) * |A|

	# Now, Dot Product = ProjectionLength * Magnitude of Vector B
	# A@B = |A| * cos(theta) * |B| = |A| * |B| * cos(theta)
	# finally, A@B = |A|*|B|*cos(theta)
	# This is how the Dot Product comes into the picture

	# But, as for our problem, we need Projection Vector.
	# we know, the Length of the Projection = |A| * cos(theta), right.
	# As for Fundamentals of Vector Algebra, Vector * Scaler = Vector, right
	# Note, Vector = Magnitude and Direction ! Important

	# here we need to find The Projection Vector
	# all we know is, Length of the Projection i.e, Magnitude of the Projection
	# now, we need to find the Projection Vector right.
	# this Magnitude of the Projection is a Scaler, as we know Scaler * Vector gives Vector i.e, our Projection Vector
	# We need to find a Vector which we can multiply with our Magnitude to get our required ProjectionVector
	# Now, the thing is, That Vector should be the Unit Vector, Unit Vector is nothing but, a Vector which has a Direction and a Magnitude of 1, right. If we Multiply a Unit Vector with a Scaler then we get a Vector which is in Unit Vector direction and having the Magnitude of Scaler Value.
	# Unit Vector of length 1 times Scaler = Vector of length Scaler Value
	# Its basic understanding, you need to understand it

	# Now, how to find a Unit Vector which will gives the required Direction ?
	# We have a Vector which is in Same Direction as the Projection right, i.e, Vector B
	# because our Projection is formed along the Vector B right, if we find a Unit Vector of Vector B, it will have the same direction of Vector B,
	# Unit Vector = B / |B|, meaning a Vector divided by its Magnitude gives a Unit Vector which is having the same direction of that Vector.
	# Now, we got the Unit Vector which can help Projection to get a Projection Vector
	# Projection Vector = Magnitude of Projection * Unit Vector along that Direction

	# Projection Vector = |A| * cos(theta) * B / |B|

	# if we want to remove the theta, then we can simply extend it to A@B
	# Projection Vector = (A@B / |B|) * (B / |B|) = (A@B / |B|**2) * B
	# and also,
	# A@B = |A|*|B|*cos(theta)
	# => B@B = |B|*|B|*cos(0) = |B|**2
	# so, Projection Vector = (A@B / B@B) * B

	# Finally, Projection Vector = (A.B / |B|**2) * B {or} = (A.B / B.B) * B

    v = np.asarray(v)
    L = np.asarray(L)
    return np.round((v.dot(L) / L.dot(L)) * L, 3)

# TC: O(n)
# SC: O(1)