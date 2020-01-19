#!/usr/bin/python3

# do pairwise triangulation to estimate local surface height, but try
# to also build up a tracking structure so we could use and refine
# this as we proceed through the matching process.

import argparse
import numpy as np

from props import getNode

from lib import project
from lib import smart
from lib import srtm

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

smart.load(proj.analysis_dir)

print('Computing pair triangulations:')
for i, i1 in enumerate(proj.image_list):
    print("doing:", i1.name)
    ned, ypr, quat = i1.get_camera_pose()
    srtm_elev = srtm.ned_interp( [ned[0], ned[1]] )
    i1_node = smart.smart_node.getChild(i1.name, True)
    i1_node.setFloat("srtm_surface_m", "%.1f" % srtm_elev)
    for j, i2 in enumerate(proj.image_list):
        if j > i:
            smart.update_surface_estimate(i1, i2)

#smart.smart_node.pretty_print()
smart.save(proj.analysis_dir)

