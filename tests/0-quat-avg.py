#!/usr/bin/python

# test of averaging several rotations together (using a quaternion
# representation)

from math import sqrt
import numpy as np
import random

import transformations

# start with the identity
#sum = transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz')
sum = np.zeros(4)
count = 0
for i in range(0,1000):
    rot = random.random()*0.25-0.125
    print "rotation =", rot
    quat = transformations.quaternion_about_axis(rot, [1, 0, 0])
    print "quat =", quat
    count += 1

    sum[0] += quat[0]
    sum[1] += quat[1]
    sum[2] += quat[2]
    sum[3] += quat[3]

    w = sum[0] / float(count)
    x = sum[1] / float(count)
    y = sum[2] / float(count)
    z = sum[3] / float(count)
    new_quat = np.array( [ w, x, y, z] )
    print "new_quat (raw) =", new_quat

    # normalize ...
    new_quat = new_quat / sqrt(np.dot(new_quat, new_quat))

    print "  avg =", new_quat
    print "  eulers =", transformations.euler_from_quaternion(new_quat, 'sxyz')
