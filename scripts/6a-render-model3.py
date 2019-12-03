#!/usr/bin/python3

# 6a-render-model3.py - investigate delauney triangulation for
# individual image surface mesh generation.

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
import cv2
import pickle
import math
import numpy as np
import os.path
import scipy.spatial

from props import getNode

from lib import groups
from lib import panda3d
from lib import project
from lib import SRTM
from lib import transformations

mesh_steps = 8                  # 1 = corners only
r2d = 180 / math.pi
tolerance = 0.5

parser = argparse.ArgumentParser(description='Set the initial camera poses.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--group', type=int, default=0, help='group index')
parser.add_argument('--texture-resolution', type=int, default=512, help='texture resolution (should be 2**n, so numbers like 256, 512, 1024, etc.')
parser.add_argument('--srtm', action='store_true', help='use srtm elevation')
parser.add_argument('--ground', type=float, help='force ground elevation in meters')
parser.add_argument('--direct', action='store_true', help='use direct pose')

args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
  
# setup SRTM ground interpolator
sss = SRTM.NEDGround( ref, 6000, 6000, 30 )

width, height = proj.cam.get_image_params()

print("Loading optimized match points ...")
matches = pickle.load( open( os.path.join(proj.analysis_dir, "matches_grouped"), "rb" ) )

# load the group connections within the image set
group_list = groups.load(proj.analysis_dir)

# initialize temporary structures for vanity stats
for image in proj.image_list:
    image.sum_values = 0.0
    image.sum_count = 0.0
    image.max_z = -9999.0
    image.min_z = 9999.0
    image.pool_xy = []
    image.pool_z = []
    image.pool_uv = []
    image.fit_xy = []
    image.fit_z = []
    image.fit_uv = []
    image.fit_edge = []

# sort through points to build a global list of feature coordinates
# and a per-image list of feature coordinates
print('Reading feature locations from optimized match points ...')
raw_points = []
raw_values = []
for match in matches:
    if match[1] == args.group and len(match[2:]) > 2:  # used by current group
        ned = match[0]
        raw_points.append( [ned[1], ned[0]] )
        raw_values.append( ned[2] )
        for m in match[2:]:
            if proj.image_list[m[0]].name in group_list[args.group]:
                image = proj.image_list[ m[0] ]
                image.pool_xy.append( [ned[1], ned[0]] )
                image.pool_z.append( -ned[2] )
                image.pool_uv.append( m[1] )
                z = -ned[2]
                image.sum_values += z
                image.sum_count += 1
                if z < image.min_z:
                    image.min_z = z
                    #print(min_z, match)
                if z > image.max_z:
                    image.max_z = z
                    #print(max_z, match)

K = proj.cam.get_K(optimized=True)
dist_coeffs = np.array(proj.cam.get_dist_coeffs(optimized=True))
def undistort(uv_orig):
    # convert the point into the proper format for opencv
    uv_raw = np.zeros((1,1,2), dtype=np.float32)
    uv_raw[0][0] = (uv_orig[0], uv_orig[1])
    # do the actual undistort
    uv_new = cv2.undistortPoints(uv_raw, K, dist_coeffs, P=K)
    # print(uv_orig, type(uv_new), uv_new)
    return uv_new[0][0]

# cull points from the per-image pool that project outside the grid boundaries
for image in proj.image_list:
    size = len(image.pool_uv)
    for i in reversed(range(len(image.pool_uv))): # iterate in reverse order
        uv_new = undistort(image.pool_uv[i])
        if uv_new[0] < 0 or uv_new[0] >= width or uv_new[1] < 0 or uv_new[1] >= height:
            print("out of range")
    
print('Generating Delaunay mesh and interpolator ...')
print(len(raw_points))
global_tri_list = scipy.spatial.Delaunay(np.array(raw_points))
interp = scipy.interpolate.LinearNDInterpolator(global_tri_list, raw_values)

def intersect2d(ned, v, avg_ground):
    p = ned[:] # copy

    # sanity check (always assume camera pose is above ground!)
    if v[2] <= 0.0:
        return p

    eps = 0.01
    count = 0
    #print("start:", p)
    #print("vec:", v)
    #print("ned:", ned)
    tmp = interp([p[1], p[0]])[0]
    if not np.isnan(tmp):
        surface = tmp
    else:
        print("Notice: starting vector intersect with avg ground elev:", avg_ground)
        surface = avg_ground
    error = abs(p[2] - surface)
    #print("p=%s surface=%s error=%s" % (p, surface, error))
    while error > eps and count < 25:
        d_proj = -(ned[2] - surface)
        factor = d_proj / v[2]
        n_proj = v[0] * factor
        e_proj = v[1] * factor
        #print(" proj = %s %s" % (n_proj, e_proj))
        p = [ ned[0] + n_proj, ned[1] + e_proj, ned[2] + d_proj ]
        #print(" new p:", p)
        tmp = interp([p[1], p[0]])[0]
        if not np.isnan(tmp):
            surface = tmp
        error = abs(p[2] - surface)
        #print("  p=%s surface=%.2f error = %.3f" % (p, surface, error))
        count += 1
    #print("surface:", surface)
    #if np.isnan(surface):
    #    #print(" returning nans")
    #    return [np.nan, np.nan, np.nan]
    dy = ned[0] - p[0]
    dx = ned[1] - p[1]
    dz = ned[2] - p[2]
    dist = math.sqrt(dx*dx+dy*dy)
    angle = math.atan2(-dz, dist) * r2d # relative to horizon
    if angle < 30:
        print(" returning high angle nans:", angle)
        return [np.nan, np.nan, np.nan]
    else:
        return p
    
def intersect_vectors(ned, v_list, avg_ground):
    pt_list = []
    for v in v_list:
        p = intersect2d(ned, v.flatten(), avg_ground)
        pt_list.append(p)
    return pt_list

for image in proj.image_list:
    if image.sum_count > 0:
        image.z_avg = image.sum_values / float(image.sum_count)
        # print(image.name, 'avg elev:', image.z_avg)
    else:
        image.z_avg = 0
    
# compute the uv grid for each image and project each point out into
# ned space, then intersect each vector with the srtm / ground /
# delauney surface.

#for group in group_list:
if True:
    group = group_list[args.group]
    #if len(group) < 3:
    #    continue
    for name in group:
        image = proj.findImageByName(name)
        print(image.name, image.z_avg)
        # scale the K matrix if we have scaled the images
        K = proj.cam.get_K(optimized=True)
        IK = np.linalg.inv(K)

        grid_list = []
        u_list = np.linspace(0, width, mesh_steps + 1)
        v_list = np.linspace(0, height, mesh_steps + 1)
        # horizontal edges
        for u in u_list:
            grid_list.append( [u, 0] )
            grid_list.append( [u, height] )
        # vertical edges (minus corners)
        for v in v_list[1:-1]:
            grid_list.append( [0, v] )
            grid_list.append( [width, v] )
        #print('grid_list:', grid_list)
        
        distorted_uv = proj.redistort(grid_list, optimized=True)
        distorted_uv = grid_list

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
        else:
            # intersect with our polygon surface approximation
            pts_ned = intersect_vectors(ned, proj_list, -image.z_avg)
            
        #print(image.name, "pts_3d (ned):\n", pts_ned)

        # convert ned to xyz and stash the result for each image
        image.grid_list = []
        for p in pts_ned:
            image.fit_xy.append([p[1], p[0]])
            image.fit_z.append(-p[2])
            image.fit_edge.append(True)
        image.fit_uv = distorted_uv
        print('len:', len(image.fit_xy), len(image.fit_z), len(image.fit_uv))

# Triangle fit algorithm
group = group_list[args.group]
#if len(group) < 3:
#    continue
for name in group:
    image = proj.findImageByName(name)
    print(image.name, image.z_avg)
    done = False
    dist_uv = []
    while not done:
        tri_list = scipy.spatial.Delaunay(np.array(image.fit_xy))
        interp = scipy.interpolate.LinearNDInterpolator(tri_list, image.fit_z)
        # find the point in the pool furthest from the triangulated surface
        next_index = None
        max_error = 0.0
        for i, pt in enumerate(image.pool_xy):
            z = interp(image.pool_xy[i])[0]
            if not np.isnan(z):
                error = abs(z - image.pool_z[i])
                if error > max_error:
                    max_error = error
                    next_index = i
        if max_error > tolerance:
            print("adding index:", next_index, "error:", max_error)
            image.fit_xy.append(image.pool_xy[next_index])
            image.fit_z.append(image.pool_z[next_index])
            image.fit_uv.append(image.pool_uv[next_index])
            image.fit_edge.append(False)
            del image.pool_xy[next_index]
            del image.pool_z[next_index]
            del image.pool_uv[next_index]
        else:
            print("finished")
            done = True
    image.fit_uv.extend(proj.undistort_uvlist(image, dist_uv))
    
    print(name, 'len:', len(image.fit_xy), len(image.fit_z), len(image.fit_uv))


# generate the panda3d egg models
dir_node = getNode('/config/directories', True)
img_src_dir = dir_node.getString('images_source')
panda3d.generate_from_fit(proj, group_list[args.group], src_dir=img_src_dir,
                          analysis_dir=proj.analysis_dir,
                          resolution=args.texture_resolution)

