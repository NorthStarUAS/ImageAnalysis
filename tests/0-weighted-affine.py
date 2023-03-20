#!/usr/bin/python

import transformations

v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
#weights = [1.0, 1.0, 1.0, 1.0]
weights = [0.1, 0.01, 0.1, 0.2]
print("original")
print(transformations.affine_matrix_from_points(v0, v1, shear=False))
print("weighted")
print(transformations.affine_matrix_from_points_weighted(v0, v1, weights, shear=False))