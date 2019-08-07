#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

import argparse
import cPickle as pickle
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os.path

sys.path.append('../lib')
import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()
proj = ProjectMgr.ProjectMgr(args.project)

print "Loading matches ..."
matches_direct = pickle.load( open( args.project + "/matches_direct", "rb" ) )
matches_sba = pickle.load( open( args.project + "/matches_sba", "rb" ) )

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
xs = []; ys = []; zs = []
max_z = 0.0
max_h = 0.0
for match in matches_direct:
    xs.append(match[0][0])
    ys.append(match[0][1])
    zs.append(-match[0][2])
    if abs(match[0][0]) > max_h: max_h = abs(match[0][0])
    if abs(match[0][1]) > max_h: max_h = abs(match[0][1])
    if -match[0][2] > max_z: max_z = -match[0][2]
ax.scatter(np.array(xs), np.array(ys), np.array(zs), label='orig (red)', c='r', marker='.')

xs = []; ys = []; zs = []
for sba in matches_sba:
    xs.append(sba[0][0])
    ys.append(sba[0][1])
    zs.append(-sba[0][2])
    if abs(sba[0][0]) > max_h: max_h = abs(sba[0][0])
    if abs(sba[0][1]) > max_h: max_h = abs(sba[0][1])
    if -sba[0][2] > max_z: max_z = -sba[0][2]
ax.scatter(np.array(xs), np.array(ys), np.array(zs), label='sba (blue)', c='b', marker='.')
ax.set_xlim([-max_h,max_h])
ax.set_ylim([-max_h,max_h])
ax.set_zlim([0,max_z*1.5])
ax.set_xlabel('North/South (m)')
ax.set_ylabel('East/West (m)')
ax.set_zlabel('Elevation (MSL)')

ax.legend(loc=0)

# xs = []; ys = []; zs = []
# for p in cam0:
#     xs.append(p[0])
#     ys.append(p[1])
#     zs.append(p[2])
# ax.scatter(np.array(xs), np.array(ys), np.array(zs), c='y', marker='^')

# xs = []; ys = []; zs = []
# for p in cam1:
#     xs.append(p[0])
#     ys.append(p[1])
#     zs.append(p[2])
# ax.scatter(np.array(xs), np.array(ys), np.array(zs), c='b', marker='^')

plt.show()

