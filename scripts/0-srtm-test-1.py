#!/usr/bin/python

# good starting point for working with dem's in python:
#   https://stevendkay.wordpress.com/2009/09/05/beginning-digital-elevation-model-work-with-python/

# source for 3 arcsec dem:
#   http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import struct
class srtmParser(object):

    def parseFile(self,filename):
        # read 1,442,401 (1201x1201) high-endian
        # signed 16-bit words into self.z
        fi=open(filename,"rb")
        contents=fi.read()
        fi.close()
        self.z=struct.unpack(">1442401H", contents)
 
    def writeCSV(self,filename):
        if self.z :
            fo=open(filename,"w")
            for row in range(0,1201):
                offset=row*1201
                thisrow=self.z[offset:offset+1201]
                rowdump = ",".join([str(z) for z in thisrow])
                fo.write("%s\n" % rowdump)
            fo.close()
        else:
            return None

from pylab import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.interpolate

if __name__ == '__main__':
    f = srtmParser()
    f.parseFile(r"N45W094.hgt")
    zzz = np.zeros((1201,1201))
    for r in range(0,1201):
        for c in range(0,1201):
            va=f.z[(1201*r)+c]
            if (va==65535 or va<0 or va>2000):
                va=0.0
            zzz[r][c]=float(va)
    # dem_pts = np.zeros((1201*1201, 2))
    # dem_vals = np.zeros((1201*1201))
    # for r in range(0,1201):
    #     for c in range(0,1201):
    #         idx = (1201*r)+c
    #         va=f.z[idx]
    #         if (va==65535 or va<0 or va>2000):
    #             va=0.0
    #         x = -94.0 + (float(c) / 1200.0)
    #         y = 45 + (float(1200-r) / 1200.0)
    #         z = va
    #         dem_pts[idx] = [x, y]
    #         dem_vals[idx] = z

    # print "constructing interpolator"
    # i = scipy.interpolate.LinearNDInterpolator(dem_pts, dem_vals)
    # print "interpolating ..."
    # print i( [-93.14530573529404, 45.220697421008396] )
 
    # logarithm color scale
    zz=np.log1p(zzz)
    imshow(zz, interpolation='bilinear',cmap=cm.gray,alpha=1.0)
    grid(False)
    show()
    
