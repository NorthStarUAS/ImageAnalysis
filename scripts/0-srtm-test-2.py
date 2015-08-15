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

srtm = SRTM.SRTM('../srtm')
srtm.parse("N45W094")
srtm.make_lla_interpolator()

#ref = [ 45.220697421008396, -93.14530573529404, 0.0 ]
#srtm.make_ned_interpolator(ref)

pts = np.array([[-93.14530573529404, 45.220697421008396], [-93.2, 45.3]], dtype=float)
result = srtm.lla_interpolate(pts)
print result
 
srtm.plot_raw()

srtm = SRTM.SRTM('../srtm')
srtm.parse("N33W111")
srtm.plot_raw()
