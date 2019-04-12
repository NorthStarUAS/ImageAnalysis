#!/usr/bin/python3

import argparse
import pickle
import os.path
from progress.bar import Bar

from props import getNode

from lib import Matcher
from lib import Pose
from lib import ProjectMgr

# working on matching features ...

parser = argparse.ArgumentParser(description='Keypoint projection.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--order', default='sequential',
                    choices=['sequential', 'fewest-matches'],
                    help='sort order')
parser.add_argument('--orient', default='relative',
                    choices=['relative', 'aircraft', 'camera', 'sba'],
                    help='yaw orientation reference')
parser.add_argument('--image', default="", help='show specific image matches')
parser.add_argument('--index', type=int, help='show specific image by index')
parser.add_argument('--direct', action='store_true', help='show matches_direct')
parser.add_argument('--sba', action='store_true', help='show matches_sba')
args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_images_info()
proj.load_features()
if args.direct:
    # recreate the pair-wise match structure
    matches_list = pickle.load( open( os.path.join(args.project, "matches_direct"), "rb" ) )
    for i1 in proj.image_list:
        i1.match_list = []
        for i2 in proj.image_list:
            i1.match_list.append([])
    for match in matches_list:
        for p1 in match[1:]:
            for p2 in match[1:]:
                if p1 == p2:
                    pass
                else:
                    i = p1[0]
                    j = p2[0]
                    image = proj.image_list[i]
                    image.match_list[j].append( [p1[1], p2[1]] )
    # for i in range(len(proj.image_list)):
    #     print(len(proj.image_list[i].match_list))
    #     print(proj.image_list[i].match_list)
    #     for j in range(len(proj.image_list)):
    #         print(i, j, len(proj.image_list[i].match_list[j]),
    #               proj.image_list[i].match_list[j])
else:
    proj.load_match_pairs(extra_verbose=False)

# lookup ned reference
ref_node = getNode("/config/ned_reference", True)
ref = [ ref_node.getFloat('lat_deg'),
        ref_node.getFloat('lon_deg'),
        ref_node.getFloat('alt_m') ]

m = Matcher.Matcher()

order = 'fewest-matches'

if args.image:
    i1 = proj.findImageByName(args.image)
    if i1 != None:
        for key in i1.match_list:
            print(key, len(i1.match_list[key]))
            if len(i1.match_list[key]):
                i2 = proj.findImageByName(key)
                print("Showing %s vs %s (%d matches)" % (i1.name, i2.name, len(i1.match_list[key])))
                status = m.showMatchOrient(i1, i2, i1.match_list[key],
                                           orient=args.orient)
    else:
        print("Cannot locate:", args.image)
elif args.index:
    i1 = proj.image_list[args.index]
    if i1 != None:
        for j, i2 in enumerate(proj.image_list):
            if len(i1.match_list[j]):
                print("Showing %s vs %s" % (i1.name, i2.name))
                status = m.showMatchOrient(i1, i2, i1.match_list[j],
                                           orient=args.orient)
    else:
        print("Cannot locate:", args.index)
elif args.order == 'sequential':
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if i2.name in i1.match_list:
                if len(i1.match_list[i2.name]):
                    print("Showing %s vs %s" % (i1.name, i2.name))
                    status = m.showMatchOrient(i1, i2, i1.match_list[i2.name],
                                               orient=args.orient)
elif args.order == 'fewest-matches':
    match_list = []
    for i, i1 in enumerate(proj.image_list):
        for j, i2 in enumerate(proj.image_list):
            if i >= j:
                # don't repeat reciprocal matches
                continue
            if len(i1.match_list[j]):
                match_list.append( ( len(i1.match_list[j]), i, j ) )
    match_list = sorted(match_list,
                        key=lambda fields: fields[0],
                        reverse=False)
    for match in match_list:
        count = match[0]
        i = match[1]
        j = match[2]
        i1 = proj.image_list[i]
        i2 = proj.image_list[j]
        print("Showing %s vs %s (matches=%d)" % (i1.name, i2.name, count))
        status = m.showMatchOrient(i1, i2, i1.match_list[j],
                                   orient=args.orient)
