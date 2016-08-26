#!/usr/bin/python

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import argparse
import commands
import cv2
import fnmatch
import os.path

sys.path.append('../lib')
import ATICorrelate
import Pose
import ProjectMgr

# for all the images in the project image_dir, detect features using the
# specified method and parameters

parser = argparse.ArgumentParser(description='Set the aircraft poses from flight data.')
parser.add_argument('--project', required=True, help='project directory')
parser.add_argument('--flight-dir', required=True, help='directory containing ATI flight data')
parser.add_argument('--shutter-latency', type=float, default=0.7, help='shutter latency from time of trigger to time of picture')

args = parser.parse_args()

proj = ProjectMgr.ProjectMgr(args.project)
proj.load_image_info()

# correlate shutter time with trigger time (based on interval
# comaparison of trigger events from the flightdata vs. image time
# stamps.)
c = ATICorrelate.Correlate()
c.load_all(args.flight_dir, args.project + "/Images")
best_correlation, best_camera_time_error = c.test_correlations()

proj.interpolateAircraftPositions(c, shutter_latency=args.shutter_latency,
                                  force=True, weight=True)

# compute the project's NED reference location (based on average of
# aircraft poses)
proj.compute_ned_reference_lla()
print "NED reference location:", proj.ned_reference_lla

proj.save()
    
