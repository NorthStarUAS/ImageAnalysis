#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import json
import math
import numpy as np

sys.path.append('../lib')
import Matcher
import ProjectMgr
import SBA
import transformations


# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()
proj.load_features()
proj.load_matches()
proj.undistort_keypoints()

m = Matcher.Matcher()

f = open(args.project + "/Matches.json", 'r')
matches_dict = json.load(f)
f.close()

sba = SBA.SBA(args.project)
sba.prepair_data( proj.image_list, matches_dict, proj.cam.get_K() )
cameras, features = sba.run()

for i, image in enumerate(proj.image_list):
    orig = image.camera_pose
    new = cameras[i]
    newq = np.array( [ new[0], new[1], new[2], new[3] ] )
    tvec = np.array( [ new[4], new[5], new[6] ] )
    Rned2cam = transformations.quaternion_matrix(newq)[:3,:3]
    cam2body = image.get_cam2body()
    Rned2body = cam2body.dot(Rned2cam)
    Rbody2ned = np.matrix(Rned2body).T
    (yaw, pitch, roll) = transformations.euler_from_matrix(Rbody2ned, 'rzyx')
    d2r = math.pi / 180.0       # a helpful constant
    #print "orig ypr =", image.camera_pose['ypr']
    #print "new ypr =", [yaw/d2r, pitch/d2r, roll/d2r]
    pos = -np.matrix(Rned2cam).T * np.matrix(tvec).T
    newned = pos.T[0].tolist()[0]
    #print "orig ned =", image.camera_pose['ned']
    #print "new ned =", newned
    image.set_camera_pose( ned=newned, ypr=[yaw/d2r, pitch/d2r, roll/d2r] )
    image.save_meta()

# update the ned coordinate in matches_dict
for i, key in enumerate(matches_dict):
    matches_dict[key]['ned'] = features[i].tolist()
    #feat = matches_dict[key]
    #ned = np.array(feat['ned'])
    #newned = features[i]
    #print "Feature %04d orig=%s new=%s" % (i, ned, newned)

# write out the updated match_dict
f = open(args.project + "/Matches.json", 'w')
json.dump(matches_dict, f, sort_keys=True)
f.close()
