#!/usr/bin/python3

import argparse
import csv
import cv2
import json
from math import atan2, isnan, pi, sqrt
import navpy
import numpy as np
import os
import pickle
import scipy.spatial

from props import getNode

from lib import camera
from lib import groups
from lib import project

# constants
r2d = 180 / pi

parser = argparse.ArgumentParser(description='Chop up an image for zooniverse.')
parser.add_argument("subjectsets", help="subject set id to local project path lookup")
parser.add_argument('subjects', help='zooniverse subjects CSV file')
parser.add_argument('classifications', help='zooniverse classifications CSV file')
parser.add_argument('--verbose', action='store_true', help='verbose')
args = parser.parse_args()

# given a path and a subject file name, find the original name this
# refers to
def find_image(path, filename):
    base, ext = os.path.splitext(filename)
    if base[-3] != "_":
        print("ERROR, filename doesn't match expected pattern:", filename)
    else:
        root = base[:-3]
        i = int(base[-2])
        j = int(base[-1])
        print(base, root, i, j)
        filel = root + ".jpg"
        fileu = root + ".JPG"
        fulll = os.path.join(path, filel)
        fullu = os.path.join(path, fileu)
        if os.path.isfile(fulll):
            return fulll, filel, i, j
        elif os.path.isfile(fullu):
            return fullu, fileu, i, j
        else:
            print("ERROR, cannot determine original file name for:",
                  path, filename)
    return None, -1, -1

no_extrapolate = False
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
    dist = sqrt(dx*dx+dy*dy)
    angle = atan2(-dz, dist) * r2d # relative to horizon
    if abs(angle) < 30:
        print(" returning high angle nans:", angle)
        return [np.nan, np.nan, np.nan]
    else:
        return p

def intersect_vectors(ned, v_list, avg_ground):
    pt_list = []
    for v in v_list:
        p = intersect2d(ned, v.flatten(), avg_ground)
        if not isnan(p[0]):
            pt_list.append(p)
    return pt_list

# build a map of subject_set_id -> local paths
subject_sets = {}
with open(args.subjectsets, 'r') as fsubjset:
    reader = csv.DictReader(fsubjset)
    for row in reader:
        id = int(row["subject_set_id"])
        subject_sets[id] = row["project_path"]

# build a map of subject id -> subject details
subjects = {}
with open(args.subjects, 'r') as fsubj:
    reader = csv.DictReader(fsubj)
    for row in reader:
        id = int(row["subject_id"])
        subject_set_id = int(row["subject_set_id"])
        meta = json.loads(row["metadata"])
        if "filename" in meta:
            if subject_set_id in subject_sets:
                #print(id, subject_set_id, meta["filename"])
                subjects[id] = { "subject_set_id": subject_set_id,
                                 "filename": meta["filename"] }
            else:
                if args.verbose:
                    print("unknown subject set id:", subject_set_id, "ignoring classsification")

by_project = {}
# traverse the classifications and do the stuff
with open(args.classifications, 'r') as fclass:
    reader = csv.DictReader(fclass)
    for row in reader:
        #print(row["classification_id"])
        #print(row["user_name"])
        #print(row["user_id"])
        #print(row["user_ip"])
        #print(row["workflow_id"])
        #print(row["workflow_name"])
        #print(row["workflow_version"])
        #print(row["created_at"])
        #print(row["gold_standard"])
        #print(row["expert"])
        #print(row["metadata"])
        #print(row["annotations"])
        #print(row["subject_data"])
        #print(row["subject_ids"])
        subject_id = int(row["subject_ids"])
        if not subject_id in subjects:
            continue
        subject_set_id = int(subjects[subject_id]["subject_set_id"])
        filename = subjects[subject_id]["filename"]
        if not subject_set_id in subject_sets:
            continue
        project_path = subject_sets[subject_set_id]
        if not project_path in by_project:
            by_project[project_path] = {}
        by_image = by_project[project_path]
        print(subject_id, subject_set_id, project_path, filename)
        fullpath, srcname, i, j = find_image(project_path, filename)
        if not srcname in by_image:
            by_image[srcname] = []
        meta = json.loads(row["metadata"])
        #print(meta)
        subject_dim = meta["subject_dimensions"][0]
        if subject_dim is None:
            continue
        print(subject_dim["naturalWidth"], subject_dim["naturalHeight"])
        subj_w = subject_dim["naturalWidth"]
        subj_h = subject_dim["naturalHeight"]
        base_w = subj_w * i
        base_h = subj_h * j
        tasks = json.loads(row["annotations"])
        task = tasks[0]
        for i, val in enumerate(task["value"]):
            print(i, val)
            x = round(val["x"])
            y = round(val["y"])
            if "r" in val:
                # palmer task
                r = round(val["r"])
            else:
                # ob task
                r = 1
                # only pass through tool 0
                if val["tool"] > 0:
                    continue
            print(x, y, r)
            deets = val["details"]
            density = deets[0]["value"]
            if len(deets) >= 2:
                confidence = deets[1]["value"]
            if len(deets) >= 3:
                comment = deets[2]["value"]
                if len(comment):
                    print("comment:", comment)
            u = base_w + x
            v = base_h + y
            by_image[srcname].append( [u, v] )

for project_path in by_project:
    print("project:", project_path)
    proj = project.ProjectMgr(project_path)
    proj.load_images_info()

    # lookup ned reference
    ref_node = getNode("/config/ned_reference", True)
    ref = [ ref_node.getFloat('lat_deg'),
            ref_node.getFloat('lon_deg'),
            ref_node.getFloat('alt_m') ]

    ned_list = []

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

    # elevation stats
    print("Computing stats...")
    tmp_list = []
    for match in matches:
        if match[1] >= 0:  # used by any group
            print("mg:", match[1])
            tmp_list.append(match[0])
    print("size of tmp_list:", len(tmp_list))
    avg = -np.mean(np.array(tmp_list)[:,2])
    median = -np.median(np.array(tmp_list)[:,2])
    std = np.std(np.array(tmp_list)[:,2])
    print("Average elevation: %.2f" % avg)
    print("Median elevation: %.2f" % median)
    print("Standard deviation: %.2f" % std)

    # sort through points
    print('Reading feature locations from optimized match points ...')
    raw_points = []
    raw_values = []
    for match in matches:
        if match[1] >= 0:           # used in a group
            ned = match[0]
            diff = abs(-ned[2] - avg)
            if diff < 10*std:
                raw_points.append( [ned[1], ned[0]] )
                raw_values.append( ned[2] )
                for m in match[2:]:
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
                print("Discarding match with excessive altitude:", match)

    print('Generating Delaunay mesh and interpolator ...')
    global_tri_list = scipy.spatial.Delaunay(np.array(raw_points))
    interp = scipy.interpolate.LinearNDInterpolator(global_tri_list, raw_values)

    for image in proj.image_list:
        if image.sum_count > 0:
            image.z_avg = image.sum_values / float(image.sum_count)
            print(image.name, 'avg elev:', image.z_avg)
        else:
            image.z_avg = 0

    K = camera.get_K(optimized=True)
    IK = np.linalg.inv(K)

    by_image = by_project[project_path]
    for srcname in sorted(by_image.keys()):
        green = (0, 255, 0)
        scale = 0.4
        print(srcname)
        pt_list = by_image[srcname]
        print(srcname, pt_list)

        # project marked points back to ned space
        base, ext = os.path.splitext(srcname)
        image = proj.findImageByName(base)
        if not image:
            continue
        print(srcname, image)

        distorted_uv = proj.redistort(pt_list, optimized=True)
        print("distorted:", distorted_uv)

        proj_list = project.projectVectors( IK,
                                            image.get_body2ned(opt=True),
                                            image.get_cam2body(),
                                            distorted_uv )
        print("proj_list:", proj_list)

        ned, ypr, quat = image.get_camera_pose(opt=True)

        # intersect with our polygon surface approximation
        pts_ned = intersect_vectors(ned, proj_list, -image.z_avg)
        print("pts_ned:", pts_ned)
        ned_list += pts_ned

        if True:
            fullpath = os.path.join(project_path, srcname)
            rgb = cv2.imread(fullpath, flags=cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH|cv2.IMREAD_IGNORE_ORIENTATION)
            for pt in pt_list:
                cv2.circle(rgb, (pt[0], pt[1]), 20, green, 5)
            preview = cv2.resize(rgb, None, fx=scale, fy=scale)
            h, w = preview.shape[:2]
            print(w, h)
            for i in range(int(w/4), w, int(w/4)):
                cv2.line(preview, (i, 0), (i, h-1), (0, 0, 0), 2)
            for i in range(int(h/4), h, int(h/4)):
                cv2.line(preview, (0, i), (w-1, i), (0, 0, 0), 2)
            cv2.imshow("debug", preview)
            cv2.waitKey()

    # stupid clustering algorithm, probably not be optimal
    max_range = 2                # meters

    print("binning:")
    bins = {}
    for ned in ned_list:
        y = int(round(ned[0]/max_range))
        x = int(round(ned[1]/max_range))
        index = "%d,%d" % (x, y)
        if index in bins:
            bins[index].append(np.array(ned))
        else:
            bins[index] = [np.array(ned)]

    for index in bins:
        sum = np.zeros(3)
        for p in bins[index]:
            sum += p
        avg = sum / len(bins[index])
        print(index, len(bins[index]), avg)

    # write out simple csv version
    filename = os.path.join(project_path, "ImageAnalysis", "zooniverse.csv")
    with open(filename, 'w') as f:
        fieldnames = ['id', 'lat_deg', 'lon_deg', 'alt_m', 'comment']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index in bins:
            sum = np.zeros(3)
            for p in bins[index]:
                sum += p
                avg = sum / len(bins[index])
            lla = navpy.ned2lla( [avg], ref[0], ref[1], ref[2] )
            tmp = {}
            tmp['id'] = index
            tmp['lat_deg'] = lla[0]
            tmp['lon_deg'] = lla[1]
            tmp['alt_m'] = lla[2]
            tmp['comment'] = "zooniverse"
            writer.writerow(tmp)
