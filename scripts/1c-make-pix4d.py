#!/usr/bin/env python3

import argparse

from lib import pose
from lib import project

parser = argparse.ArgumentParser(description='Create a pix4d.csv file for a folder of geotagged images.')
parser.add_argument('project', help='project directory')
parser.add_argument('--force-altitude', type=float, help='Fudge altitude geotag for stupid dji phantom 4 pro v2.0')
parser.add_argument('--force-heading', type=float, help='Force heading for every image')
parser.add_argument('--yaw-from-groundtrack', action='store_true', help='estimate yaw angle from ground track')
args = parser.parse_args()

image_dir = args.project

proj = project.ProjectMgr(args.project)

pose.make_pix4d(image_dir, args.force_altitude, args.force_heading, args.yaw_from_groundtrack)
