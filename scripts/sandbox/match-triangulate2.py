#!/usr/bin/python3

# do pairwise triangulation to estimate local surface height, but try
# to also build up a tracking structure so we could use and refine
# this as we proceed through the matching process.

import argparse
import numpy as np

from props import getNode

from lib import project
from lib import srtm
from lib import surface

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
# proj.load_features(descriptors=False)
proj.load_match_pairs()

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]
# setup SRTM ground interpolator
srtm.initialize( ref, 6000, 6000, 30 )

print('Computing pair triangulations:')
for i, i1 in enumerate(proj.image_list):
    tri_node = i1.node.getChild("tri_surface_pairs", True)
    sum = 0.0
    count = 0
    for j, i2 in enumerate(proj.image_list):
        if j == i:
            continue

        # srtm based elevation
        ned1, ypr1, quat1 = i1.get_camera_pose()
        ned2, ypr2, quat2 = i2.get_camera_pose()
        g1 = srtm.ned_interp( [ned1[0], ned1[1]] )
        g2 = srtm.ned_interp( [ned2[0], ned2[1]] )

        # pose/triangulation based elevation
        points = surface.triangulate_ned(i1, i2)
        if not points is None:
            pair_node = tri_node.getChild(i2.name, True)
            pair_node.setFloat("surface_m", float("%.1f" % -np.average(points[2])))
            pair_node.setInt("weight", points.shape[1])
            pair_node.setFloat("stddev", float("%.1f" % np.std(points[2])))
            num_matches = points.shape[1]
            sum += np.average(points[2])*num_matches
            count += num_matches
            print(" ", i1.name, "+", i2.name, "srtm: %.1f" % ((g1 + g2)*0.5), "triang est: %.1f" % np.average(points[2]), "triang std: %.1f" % np.std(points[2]))
    sum = 0
    count = 0
    for child in tri_node.getChildren():
        pair_node = tri_node.getChild(child)
        surf = pair_node.getFloat("surface_m")
        weight = pair_node.getInt("weight")
        stddev = pair_node.getFloat("stddev")
        if stddev < 25:
            sum += surf * weight
            count += weight
    if count > 0:
        print(i1.name, "estimated surface below:", "%.1f" % (sum / count))
        i1.node.setFloat("tri_surface_m", float("%.1f" % (sum / count)))
    else:
        print(i1.name, "no matches, no triangulation, no estimate")

# proj.save_images_info()
