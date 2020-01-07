#!/usr/bin/python3

import argparse
import math
import os

from auracore import wgs84      # github.com/AuraUAS/aura-core
from props import getNode

from lib import pose
from lib import project

d2r = math.pi / 180.0
r2d = 180.0 / math.pi

parser = argparse.ArgumentParser(description='Create a pix4d.csv file for a folder of geotagged images.')
parser.add_argument('project', help='project directory')
parser.add_argument('--force-altitude', type=float, help='Fudge altitude geotag for stupid dji phantom 4 pro v2.0')
parser.add_argument('--force-heading', type=float, help='Force heading for every image')
parser.add_argument('--yaw-from-groundtrack', action='store_true', help='estimate yaw angle from ground track')
args = parser.parse_args()

image_dir = args.project

proj = project.ProjectMgr(args.project)

pose.make_pix4d(image_dir, args.force_altitude, args.force_heading, args.yaw_from_groundtrack)
