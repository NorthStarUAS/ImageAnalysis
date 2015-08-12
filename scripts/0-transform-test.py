#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import math
import numpy as np
import os.path

sys.path.append('../lib')
import Image
import Pose
import ProjectMgr
import transformations

d2r = math.pi / 180.0

Rz = transformations.rotation_matrix(90*d2r, [0, 0, 1])
print Rz
Ry = transformations.rotation_matrix(-90*d2r, [0, 1, 0])
print Ry

print Ry.dot(Rz)
