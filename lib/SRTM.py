# a class to manage SRTM surfaces

import json
import numpy as np
import os
from pylab import *
import random
import scipy.interpolate
import struct
import urllib
import zipfile

import navpy

class SRTM():
    def __init__(self, dict_path):
        self.srtm_dict = {}
        self.srtm_cache_dir = '/var/tmp' # unless set otherwise
        self.srtm_z = None
        self.i = None
        self.load_srtm_dict(dict_path)
        
    # load the directory download dictionary (mapping a desired file
    # to a download path.)
    def load_srtm_dict(self, dict_path):
        dict_file = dict_path + '/srtm.json'
        try:
            f = open(dict_file, 'r')
            self.srtm_dict = json.load(f)
            f.close()
        except:
            print "Notice: unable to read =", dict_file

    # if we'd like a persistant place to cache srtm files so they don't
    # need to be re-download every time we run anything
    def set_srtm_cache_dir(self, srtm_cache_dir):
        if not os.path.exists(srtm_cache_dir):
            print "Notice: srtm cache path doesn't exist =", srtm_cache_dir
        else:
            self.srtm_cache_dir = srtm_cache_dir

    # download and extract srtm file into cache directory
    def download_srtm(self, fileroot):
        if fileroot in self.srtm_dict:
            url = self.srtm_dict[fileroot]
            download_file = self.srtm_cache_dir + '/' + fileroot + '.hgt.zip'
            print "Notice: downloading:", url
            file = urllib.URLopener()
            file.retrieve(url, download_file)
            print "Exracting:", download_file
            zip = zipfile.ZipFile(download_file)
            zip.extractall(self.srtm_cache_dir)
            os.remove(download_file)
            return True
        else:
            print "Notice: requested srtm that is outside catalog"
            return False
        
    def parse(self, fileroot):
        cache_file = self.srtm_cache_dir + '/' + fileroot + '.hgt'
        if not os.path.exists(cache_file):
            if not self.download_srtm(fileroot):
                return

        print "Notice: parsing SRTM file:", cache_file
        # read 1,442,401 (1201x1201) high-endian
        # signed 16-bit words into self.z
        f = open(cache_file, "rb")
        contents = f.read()
        f.close()
        self.srtm_z = struct.unpack(">1442401H", contents)

    def make_lla_interpolator(self):
        print "Notice: constructing lla interpolator"
        
        srtm_pts = np.zeros((1201, 1201))
        for r in range(0,1201):
            for c in range(0,1201):
                idx = (1201*r)+c
                va = self.srtm_z[idx]
                if va == 65535 or va < 0 or va > 10000:
                    va = 0.0
                z = va
                srtm_pts[c,1200-r] = z
        x = np.linspace(-94.0, -93.0, 1201)
        y = np.linspace(45.0, 46.0, 1201)
        self.lla_interp = scipy.interpolate.RegularGridInterpolator((x, y), srtm_pts)
        #print self.i([-93.14530573529404, 45.220697421008396])
        #for i in range(20):
        #    x = -94.0 + random.random()
        #    y = 45 + random.random()
        #    lnd = [0] # self.i([x,y])
        #    mif = my_interpolating_function([x,y])
        #    print "lnd=%s, mif=%s" % (lnd, mif)

    # notice: this is *really* slow.  Adding the lla2ned conversion
    # makes even more slow.  I think what I want to do is build as
    # many SRTM instances as we need to cover an area, then resample
    # just the needed area, at some reasonable grid size, convert just
    # the resampled points to ned, and build the interpolator out of
    # the subset of points.  A DEM file can cover 60nm in latitude and
    # 60nm*cos(lat) in the longitude direction.
    
    def make_ned_interpolator(self, ref):
        print "Notice: constructing ned interpolator"

        # The LinearNDInterpolator works well, but is slow to
        # construct, ned data is not 'regularly' gridded so we need to
        # use this fancier interpolator
        ned_pts = np.zeros((1201*1201, 2))
        ned_vals = np.zeros((1201*1201))
        for r in range(0,1201):
            print "%.1f%%" % (float(r)/float(12.01))
            for c in range(0,1201):
                idx = (1201*r)+c
                va = self.srtm_z[idx]
                if va == 65535 or va < 0 or va > 10000:
                    va = 0.0
                lon = -94.0 + (float(c) / 1200.0)
                lat = 45 + (float(1200-r) / 1200.0)
                alt = va
                ned = navpy.lla2ned( lat, lon, alt, ref[0], ref[1], ref[2] )
                ned_pts[idx] = ned[0:2]
                ned_vals[idx] = ned[2]
        self.ned_interp = scipy.interpolate.LinearNDInterpolator(ned_pts, ned_vals)
        
    def lla_interpolate(self, point_list):
        return self.lla_interp(point_list)

    def ned_interpolate(self, point_list):
        return self.ned_interp(point_list)

    def plot_raw(self):
        zzz = np.zeros((1201,1201))
        for r in range(0,1201):
            for c in range(0,1201):
                va=self.srtm_z[(1201*r)+c]
                if (va==65535 or va<0 or va>10000):
                    va=0.0
                zzz[r][c]=float(va)
 
        #zz=np.log1p(zzz)
        imshow(zzz, interpolation='bilinear',cmap=cm.gray,alpha=1.0)
        grid(False)
        show()

