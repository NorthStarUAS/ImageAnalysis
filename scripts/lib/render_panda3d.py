#!/usr/bin/python3

import math
import numpy as np
import os
import pickle
import scipy.spatial

from props import getNode

from . import camera
from .logger import log, qlog
from . import panda3d
from . import project
#from . import objmtl            # temporary?

r2d = 180 / math.pi

grid_steps = 8
texture_resolution = 512
use_direct_pose = False
force_ground_elevation_m = None
use_srtm_surface = None
no_extrapolate = False

def intersect2d(interp, ned, v, avg_ground):
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
    if no_extrapolate or not np.isnan(tmp):
        surface = tmp
    else:
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
        if no_extrapolate or not np.isnan(tmp):
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
    
def intersect_vectors(interp, ned, v_list, avg_ground):
    pt_list = []
    for v in v_list:
        p = intersect2d(interp, ned, v.flatten(), avg_ground)
        pt_list.append(p)
    return pt_list

def build_map(proj, group_list, group_index):
    # lookup ned reference
    ref_node = getNode("/config/ned_reference", True)
    ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]

    log("Loading optimized match points ...")
    matches = pickle.load( open( os.path.join(proj.analysis_dir, "matches_grouped"), "rb" ) )

    # initialize temporary structures for vanity stats
    for image in proj.image_list:
        image.sum_values = 0.0
        image.sum_count = 0.0
        image.max_z = -9999.0
        image.min_z = 9999.0

    # elevation stats
    log("Computing stats...")
    ned_list = []
    for match in matches:
        if match[1] == group_index:  # used by current group
            ned_list.append(match[0])
    avg = -np.mean(np.array(ned_list)[:,2])
    std = np.std(np.array(ned_list)[:,2])
    log("Average elevation: %.2f" % avg)
    log("Standard deviation: %.2f" % std)

    # sort through points
    log('Reading feature locations from optimized match points ...')
    raw_points = []
    raw_values = []
    for match in matches:
        if match[1] == group_index:  # used by current group
            ned = match[0]
            diff = abs(-ned[2] - avg)
            if diff < 10*std:
                raw_points.append( [ned[1], ned[0]] )
                raw_values.append( ned[2] )
                for m in match[2:]:
                    if proj.image_list[m[0]].name in group_list[group_index]:
                        image = proj.image_list[ m[0] ]
                        z = -ned[2]
                        image.sum_values += z
                        image.sum_count += 1
                        if z < image.min_z:
                            image.min_z = z
                            #print(min_z, match)
                        if z > image.max_z:
                            image.max_z = z
                            #print(max_z, match)
            else:
                log("Discarding match with excessive altitude:", match)

    # save the surface definition as a separate file
    models_dir = os.path.join(proj.analysis_dir, 'models')
    if not os.path.exists(models_dir):
        log("Notice: creating models directory =", models_dir)
        os.makedirs(models_dir)
    surface = { 'points': raw_points,
                'values': raw_values }
    pickle.dump(surface, open(os.path.join(proj.analysis_dir, 'models', 'surface.bin'), "wb"))

    log('Generating Delaunay mesh and interpolator ...')
    global_tri_list = scipy.spatial.Delaunay(np.array(raw_points))
    interp = scipy.interpolate.LinearNDInterpolator(global_tri_list, raw_values)


    for image in proj.image_list:
        if image.sum_count > 0:
            image.z_avg = image.sum_values / float(image.sum_count)
            # log(image.name, 'avg elev:', image.z_avg)
        else:
            image.z_avg = 0

    # compute the uv grid for each image and project each point out into
    # ned space, then intersect each vector with the srtm / ground /
    # delauney surface.

    #for group in group_list:
    if True:
        group = group_list[group_index]
        #if len(group) < 3:
        #    continue
        for name in group:
            image = proj.findImageByName(name)
            log(image.name, image.z_avg)
            width, height = camera.get_image_params()
            # scale the K matrix if we have scaled the images
            K = camera.get_K(optimized=True)
            IK = np.linalg.inv(K)

            grid_list = []
            u_list = np.linspace(0, width, grid_steps + 1)
            v_list = np.linspace(0, height, grid_steps + 1)
            #print "u_list:", u_list
            #print "v_list:", v_list
            for v in v_list:
                for u in u_list:
                    grid_list.append( [u, v] )
            #print 'grid_list:', grid_list
            image.distorted_uv = proj.redistort(grid_list, optimized=True)

            if use_direct_pose:
                proj_list = project.projectVectors( IK, image.get_body2ned(),
                                                    image.get_cam2body(),
                                                    grid_list )
            else:
                #print(image.get_body2ned(opt=True))
                proj_list = project.projectVectors( IK, image.get_body2ned(opt=True),
                                                    image.get_cam2body(),
                                                    grid_list )
            #print 'proj_list:', proj_list

            if use_direct_pose:
                ned, ypr, quat = image.get_camera_pose()
            else:
                ned, ypr, quat = image.get_camera_pose(opt=True)
            #print('cam orig:', image.camera_pose['ned'], 'optimized:', ned)
            if force_ground_elevation_m:
                pts_ned = project.intersectVectorsWithGroundPlane(ned, force_ground_elevation_m, proj_list)
            elif use_srtm_surface:
                # setup SRTM ground interpolator
                from lib import srtm
                srtm.initialize( ref, 6000, 6000, 30 )
                pts_ned = srtm.interpolate_vectors(ned, proj_list)
            elif False:
                # this never seemed that productive
                print(image.name, image.z_avg)
                pts_ned = project.intersectVectorsWithGroundPlane(ned, image.z_avg, proj_list)
            elif True:
                # intersect with our polygon surface approximation
                pts_ned = intersect_vectors(interp, ned, proj_list, -image.z_avg)
            elif False:
                # (moving away from the binned surface approach in this
                # script towards the above delauney interpolation
                # approach)
                # intersect with 2d binned surface approximation
                pts_ned = bin2d.intersect_vectors(interp, ned, proj_list, -image.z_avg)

            #print(image.name, "pts_3d (ned):\n", pts_ned)

            # convert ned to xyz and stash the result for each image
            image.grid_list = []
            for p in pts_ned:
                image.grid_list.append( [p[1], p[0], -p[2]] )

    # generate the panda3d egg models
    dir_node = getNode('/config/directories', True)
    img_src_dir = dir_node.getString('images_source')
    panda3d.generate_from_grid(proj, group_list[group_index],
                               src_dir=img_src_dir,
                               analysis_dir=proj.analysis_dir,
                               resolution=texture_resolution)

    # and the obj/mtl 3d format (hopefully for webgl use)
    #objmtl.generate_from_grid(proj, group_list[group_index],
    #                           src_dir=img_src_dir,
    #                           analysis_dir=proj.analysis_dir,
    #                           resolution=texture_resolution)
    
    # call the ac3d generator
    # AC3D.generate(proj.image_list, group_list[0], src_dir=img_src_dir,
    #               analysis_dir=proj.analysis_dir, base_name='direct',
    #               version=1.0, trans=0.1, resolution=args.texture_resolution)
