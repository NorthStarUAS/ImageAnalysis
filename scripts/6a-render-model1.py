#!/usr/bin/python3

# for all the images in the fitted group, generate a 2d polygon
# surface fit.  Then project the individual images onto this surface
# and generate an AC3D model.
#
# Note: insufficient image overlap (or long linear image match chains)
# are not good.  Ideally we would have a nice mesh of match pairs for
# best results.
#
# this script can also project onto the SRTM surface, or a flat ground
# elevation plane.

import argparse
import pickle
import numpy as np
import os.path
import sys

from props import getNode

sys.path.append('../lib')
import AC3D
import Groups
import Panda3d
import Pose
import ProjectMgr
import SRTM
import transformations

import match_culling as cull

ac3d_steps = 8

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--bins', type=int, default=128, help='surface bins for stats')
parser.add_argument('--texture-resolution', type=int, default=512, help='texture resolution (should be 2**n, so numbers like 256, 512, 1024, etc.')
parser.add_argument('--srtm', action='store_true', help='use srtm elevation')
parser.add_argument('--ground', type=float, help='force ground elevation in meters')
parser.add_argument('--direct', action='store_true', help='use direct pose')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
proj.undistort_keypoints(optimized=True)

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
  
# setup SRTM ground interpolator
sss = SRTM.NEDGround( ref, 6000, 6000, 30 )

print("Loading optimized match points ...")
matches_opt = pickle.load( open( os.path.join(args.project, "matches_opt"), "rb" ) )

# load the group connections within the image set
groups = Groups.load(args.project)

# testing ...
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def intersect2d(ned, v, m):
    p = ned[:] # copy hopefully

    # sanity check (always assume camera pose is above ground!)
    if v[2] <= 0.0:
        return p

    eps = 0.01
    count = 0
    #print("start:", p)
    #print("vec:", v)
    #print("ned:", ned)
    surface = polyval2d(p[0], p[1], m)
    error = abs(p[2] - surface)
    #print("  p=%s surface=%s error=%s" % (p, surface, error))
    while error > eps and count < 25 and surface <= 0:
        d_proj = -(ned[2] - surface)
        factor = d_proj / v[2]
        n_proj = v[0] * factor
        e_proj = v[1] * factor
        #print("proj = %s %s" % (n_proj, e_proj))
        p = [ ned[0] + n_proj, ned[1] + e_proj, ned[2] + d_proj ]
        #print("new p:", p)
        surface = polyval2d(p[0], p[1], m)
        error = abs(p[2] - surface)
        #print("  p=%s surface=%.2f error = %.3f" % (p, surface, error))
        count += 1
    #print("surface:", p)
    if surface <= 100000:
        return p
    else:
        #print " returning nans"
        return np.zeros(3)*np.nan
    
def intersect_vectors(ned, v_list, m):
    pt_list = []
    for v in v_list:
        p = intersect2d(ned, v.flatten(), m)
        pt_list.append(p)
    return pt_list

import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
# first determine surface elevation stats so we can discard outliers
z = []
for match in matches_opt:
    count = 0
    for m in match[1:]:
        if m[0] in groups[0]:
            count += 1
    if count >= 3:
        ned = match[0]
        z.append(ned[2])
zavg = np.mean(z)
zstd = np.std(z)
print('elevation stats:', zavg, zstd)

# now build the surface
xfit = []
yfit = []
zfit = []
for match in matches_opt:
    count = 0
    for p in match[1:]:
        if p[0] in groups[0]:
            count += 1
    if count >= 3:
        ned = match[0]
        d = abs(ned[2] - zavg)
        if True or d <= 2*zstd:
            xfit.append(ned[0])
            yfit.append(ned[1])
            zfit.append(ned[2])
        #else:
        #    # mark this elevation unused
        #    match[0][2] = None
xfit = np.array(xfit)
yfit = np.array(yfit)
zfit = np.array(zfit)
plt.figure()
# flip from NED to XYZ
plt.scatter(yfit, xfit, 100, -zfit, cmap=cm.jet)
plt.colorbar()
plt.title("Sparsely sampled function.")

# Fit a n order, 2d polynomial
m = polyfit2d(xfit, yfit, zfit, order=1)

# test fit
znew = polyval2d(xfit, yfit, m)
#for i in range(len(znew)):
#    print(polyval2d(xfit[i], yfit[i], m))
#    print('z:', zfit[i], znew[i], zfit[i] - znew[i])
plt.figure()
# flip from NED to XYZ
plt.scatter(yfit, xfit, 100, -znew, cmap=cm.jet)
plt.colorbar()
plt.title("Approximation function.")

import binned_surface
bin2d = binned_surface.binned_surface()
bin2d.make(xfit, yfit, zfit, bins=args.bins)
bin2d.fill()

if False:
    # test surface approximation fit
    print('size:', xfit.size)
    for i in range(xfit.size):
        za = bin2d.query(xfit[i], yfit[i])
        e = za - zfit[i]
        print(i, e)
               
plt.figure()
#plt.pcolormesh(xedges, yedges, stats, cmap=cm.jet)
#plt.colorbar()
#print(mean)
#plt.pcolormesh(xedges[1:], yedges[1:], stats)
plt.imshow(-bin2d.mean, origin='lower', cmap=cm.jet)
plt.colorbar()
plt.title("Binned data.")

# test scipy.interpolate.Rbf() (Uses LOTS of memory if the data set is
# anything but tiny!)
test_rbf = False
if test_rbf:
    import scipy.interpolate
    #f = scipy.interpolate.interp2d(xfit, yfit, zfit, kind='cubic')
    f = scipy.interpolate.Rbf(xfit, yfit, zfit, smooth=3.0)
    # test fit
    znew = f(xfit, yfit)
    plt.figure()
    # flip from NED to XYZ
    plt.scatter(yfit, xfit, 100, -znew, cmap=cm.jet)
    plt.colorbar()
    plt.title("radial basis functions.")

# show any plots
plt.show()

# compute the uv grid for each image and project each point out into
# ned space, then intersect each vector with the srtm / ground /
# polynomial surface.

# for each image, find all the placed features, and compute an average
# elevation
for image in proj.image_list:
    image.z_list = []
    image.grid_list = []
for i, match in enumerate(matches_opt):
    count = 0
    for p in match[1:]:
        if p[0] in groups[0]:
            count += 1
    if count >= 2:
        # in the solution if a match connects to at least two images
        # in the primary group
        for m in match[1:]:
            ned = match[0]
            index = m[0]
            #print('index:', index)
            if index in groups[0]:
                proj.image_list[index].z_list.append(-ned[2])
for image in proj.image_list:
    if len(image.z_list):
        avg = np.mean(np.array(image.z_list))
        std = np.std(np.array(image.z_list))
    else:
        avg = 0.0
        std = 0.0
    image.z_avg = avg
    image.z_std = std
    # print(image.name, 'features:', len(image.z_list), 'avg:', avg, 'std:', std)

# for fun rerun through the matches and find elevation outliers
outliers = []
for i, match in enumerate(matches_opt):
    ned = match[0]
    error_sum = 0
    for m in match[1:]:
        if m[0] in groups[0]:
            image = proj.image_list[m[0]]
            dist = abs(-ned[2] - image.z_avg)
            error_sum += dist
    if error_sum > 3 * (image.z_std * len(match[1:])):
        # print('possible outlier match index:', i, error_sum, 'z:', ned[2])
        outliers.append( [error_sum, i] )

result = sorted(outliers, key=lambda fields: fields[0], reverse=True)
for line in result:
    #print('index:', line[1], 'error:', line[0])
    #cull.draw_match(line[1], 1, matches_opt, proj.image_list)
    pass
    
depth = 0.0
#for group in groups:
if True:
    group = groups[0]
    #if len(group) < 3:
    #    continue
    for g in group:
        image = proj.image_list[g]
        print(image.name, image.z_avg)
        width, height = image.get_size()
        # scale the K matrix if we have scaled the images
        K = proj.cam.get_K(optimized=True)
        IK = np.linalg.inv(K)

        grid_list = []
        u_list = np.linspace(0, width, ac3d_steps + 1)
        v_list = np.linspace(0, height, ac3d_steps + 1)
        #print "u_list:", u_list
        #print "v_list:", v_list
        for v in v_list:
            for u in u_list:
                grid_list.append( [u, v] )
        #print 'grid_list:', grid_list
        image.distorted_uv = proj.redistort(grid_list, optimized=True)

        if args.direct:
            proj_list = proj.projectVectors( IK, image.get_body2ned(),
                                             image.get_cam2body(), grid_list )
        else:
            #print(image.get_body2ned(opt=True))
            proj_list = proj.projectVectors( IK, image.get_body2ned(opt=True),
                                             image.get_cam2body(), grid_list )
        #print 'proj_list:', proj_list

        if args.direct:
            ned, ypr, quat = image.get_camera_pose()
        else:
            ned, ypr, quat = image.get_camera_pose(opt=True)
        #print('cam orig:', image.camera_pose['ned'], 'optimized:', ned)
        if args.ground:
            pts_ned = proj.intersectVectorsWithGroundPlane(ned,
                                                           args.ground, proj_list)
        elif args.srtm:
            pts_ned = sss.interpolate_vectors(ned, proj_list)
        elif False:
            # this never seemed that productive
            print(image.name, image.z_avg)
            pts_ned = proj.intersectVectorsWithGroundPlane(ned,
                                                           image.z_avg,
                                                           proj_list)
        elif False:
            # broke/fixme?
            # intersect with our polygon surface approximation
            pts_ned = intersect_vectors(ned, proj_list, m)
        else:
            # intersect with 2d binned surface approximation
            pts_ned = bin2d.intersect_vectors(ned, proj_list, -image.z_avg)
            
        #print "pts_3d (ned):\n", pts_ned

        # convert ned to xyz and stash the result for each image
        image.grid_list = []
        ground_sum = 0
        for p in pts_ned:
            image.grid_list.append( [p[1], p[0], -(p[2]+depth)] )
            #image.grid_list.append( [p[1], p[0], -(depth)] )
            ground_sum += -p[2]
        depth -= 0.01                # favor last pictures above earlier ones

# generate the panda3d egg models
dir_node = getNode('/config/directories', True)
img_src_dir = dir_node.getString('images_source')
Panda3d.generate(proj.image_list, groups[0], src_dir=img_src_dir,
                 project_dir=args.project, base_name='direct',
                 version=1.0, trans=0.1, resolution=args.texture_resolution)

# call the ac3d generator
AC3D.generate(proj.image_list, groups[0], src_dir=img_src_dir,
              project_dir=args.project, base_name='direct',
              version=1.0, trans=0.1, resolution=args.texture_resolution)

if not args.ground:
    print('Avg ground elevation (SRTM):', ground_sum / len(pts_ned))
