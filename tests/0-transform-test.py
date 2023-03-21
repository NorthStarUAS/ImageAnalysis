#!/usr/bin/python

from math import pi

from transformations import rotation_matrix

d2r = pi / 180.0

Rz = rotation_matrix(-90*d2r, [0, 0, 1])
print(Rz)
Ry = rotation_matrix(-90*d2r, [0, 1, 0])
print(Ry)

print(Ry.dot(Rz))

print()

Ry = rotation_matrix(180*d2r, [0, 1, 0])
print(Ry)
Rz = rotation_matrix(-90*d2r, [0, 0, 1])
print(Rz)
print(Ry.dot(Rz))
