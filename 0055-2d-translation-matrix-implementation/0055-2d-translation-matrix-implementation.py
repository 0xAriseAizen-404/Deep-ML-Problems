import numpy as np
def translate_object(points, tx, ty):
    translation_values = [tx, ty]
    transformation_matrix = np.eye(len(translation_values) + 1)
    for axis_index in range(len(translation_values)):
        transformation_matrix[axis_index][-1] = translation_values[axis_index]

    translated_points = []
    for point in points:
        homogeneous_point = np.append(point, 1)
        transformed_point = transformation_matrix @ homogeneous_point
        transformed_point = transformed_point.tolist()
        transformed_point.pop()  # Remove homogeneous coordinate
        translated_points.append(transformed_point)
    return translated_points