#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

from math import pi
import numpy as np

sys.path.append('../lib')
import Image
import Pose
import ProjectMgr
import transformations

d2r = pi / 180.0

Rz = transformations.rotation_matrix(-90*d2r, [0, 0, 1])
print Rz
Ry = transformations.rotation_matrix(-90*d2r, [0, 1, 0])
print Ry

print Ry.dot(Rz)

print

Ry = transformations.rotation_matrix(180*d2r, [0, 1, 0])
print Ry
Rz = transformations.rotation_matrix(-90*d2r, [0, 0, 1])
print Rz
print Ry.dot(Rz)
