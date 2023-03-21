#!/usr/bin/python

from transformations import affine_matrix_from_points, affine_matrix_from_points_weighted

v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
#weights = [1.0, 1.0, 1.0, 1.0]
weights = [0.1, 0.01, 0.1, 0.2]
print("original")
print(affine_matrix_from_points(v0, v1, shear=False))
print("weighted")
print(affine_matrix_from_points_weighted(v0, v1, weights, shear=False))