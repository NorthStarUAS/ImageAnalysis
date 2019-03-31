#!/usr/bin/python3

# As the number of images increase linearly, required resources and
# processing time can increase exponentially.  This script partitions
# an image set into smaller regional groups allowing each partition to
# be processed individually.  This is a concession due to the lack of
# infinite resources.

import argparse
import math
import os
import pickle

from lib import ProjectMgr

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--size', default=3000, help='target image group size')
parser.add_argument('--pad', type=float, default=1.05, help='pad cell radius to grab more neighbors')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()

# compute bounds (x, y) for entire image set.  At the same time, make
# a list of x & y coords so we can later find dividing partions that
# attempt to balance group sizes by number of images, not area
# covered.  The result will be imperfect, but if the operator flew an
# even rectangular grid, this approach should work pretty well.

bounds = None
xlist = []
ylist = []
for image in proj.image_list:
    ned, ypr, quat = image.get_camera_pose()
    image.is_assigned = False
    if bounds is None:
        bounds = [ [ned[1], ned[1]], [ned[0], ned[0]] ]
    else:
        if ned[1] < bounds[0][0]: bounds[0][0] = ned[1]
        if ned[1] > bounds[0][1]: bounds[0][1] = ned[1]
        if ned[0] < bounds[1][0]: bounds[1][0] = ned[0]
        if ned[0] > bounds[1][1]: bounds[1][1] = ned[0]
    xlist.append( ned[1] )
    ylist.append( ned[0] )

xlist = sorted(xlist)
ylist = sorted(ylist)
size = len(proj.image_list)

print('image set size:', size)
print('bounds:', bounds)

dx = bounds[0][1] - bounds[0][0]
dy = bounds[1][1] - bounds[1][0]
print('dx: %.2f' % dx, 'dy: %.2f' % dy)

divx = 1
divy = 1

done = False
while not done:
    # determine the cell division points to maintain even image number
    # distribution
    xsplits = []
    di = int(size / divx)
    for i in range(divx):
        xsplits.append( xlist[i*di] )
    xsplits.append( xlist[-1] )
    print('xsplits:', xsplits)
    
    ysplits = []
    dj = int(size / divy)
    for j in range(divy):
        ysplits.append( ylist[j*dj] )
    ysplits.append( ylist[-1] )
    print('ysplits:', ysplits)

    # compute the bounds of each area and then count the images that
    # will fall into it
    done = True
    for j in range(divy):
        for i in range(divx):
            images = []
            dx = xsplits[i+1] - xsplits[i]
            dy = ysplits[j+1] - ysplits[j]
            xcenter = xsplits[i] + dx * 0.5
            ycenter = ysplits[j] + dy * 0.5
            radius = math.sqrt( 0.25*dx*dx + 0.25*dy*dy)
            print('cell:', i, j, 'size: %.2f x %.2f' % (dx, dy), "%.2f" % xcenter, "%.2f" % ycenter, 'radius: %.2f' % radius)
            for image in proj.image_list:
                ned, ypr, quat = image.get_camera_pose()
                x = xcenter - ned[1]
                y = ycenter - ned[0]
                dist = math.sqrt(x*x + y*y)
                if dist <= radius * args.pad:
                    images.append(image.name)
                    image.is_assigned = True
            print('  images in cell:', len(images))
            dirname = os.path.join(proj.analysis_dir, 'area-%d%d' % (i, j))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            pickle.dump(images, open(os.path.join(dirname, "image_list"), 'wb'))
            if len(images) > float(args.size) * args.pad:
                done = False

    if not done:
        # figure out which dimension to subdivide to maintain squarish cells
        if dx / divx > dy / divy:
            divx += 1
        else:
            divy += 1
        print('divs:', divx, divy)

        
for image in proj.image_list:
    if not image.is_assigned:
        print(image.name, 'not assigned to any group.')
