#!/usr/bin/env python3

# use graphviz library to generate a visualization of the image (feature)
# connectivity in the image set.

import argparse
import pickle
import os.path

import pygraphviz as pgv

from lib import project

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('project', help='project directory')
args = parser.parse_args()

proj = project.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
proj.undistort_keypoints()

# no! (maybe?)
print("Loading direct matches...")
matches = pickle.load( open( os.path.join(args.project, 'matches_direct'), 'rb' ) )

#print("Loading grouped matches...")
#matches = pickle.load( open( os.path.join(args.project, 'matches_grouped'), 'rb' ) )
#print("features:", len(matches))

A = pgv.AGraph()

for match in matches:
    for m1 in match[1:]:
        for m2 in match[1:]:
            if m1 == m2:
                continue
            i1 = proj.image_list[m1[0]]
            i2 = proj.image_list[m2[0]]
            A.add_edge(i1.name, i2.name)

A.layout()

# i like prog='dot' best so far (circle might be useful)
A.draw( os.path.join(args.project, 'graphvis.pdf'), prog='dot' )
