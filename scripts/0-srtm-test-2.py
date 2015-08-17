#!/usr/bin/python

# good starting point for working with dem's in python:
#   https://stevendkay.wordpress.com/2009/09/05/beginning-digital-elevation-model-work-with-python/

# source for 3 arcsec dem:
#   http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import numpy as np

sys.path.append('../lib')
import SRTM

#lla_ref = [ 45.220697421008396, -93.14530573529404, 0.0 ]
lla_ref = [ 33.220697421008396, -110.14530573529404, 0.0 ]
sss = SRTM.NEDGround( lla_ref, 100000, 100000, 75 )

