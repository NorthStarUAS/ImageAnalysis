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

# return the lower left corner of the 1x1 degree tile containing
# the specified lla coordinate
def lla_ll_corner(lat_deg, lon_deg):
    return int(floor(lat_deg)), int(floor(lon_deg))

# return the tile base name for the specified coordinate
def make_tile_name(lat, lon):
    ll_lat, ll_lon = lla_ll_corner(lat, lon)
    if ll_lat < 0:
        slat = "S%2d" % ll_lat
    else:
        slat = "N%2d" % ll_lat
    if ll_lon < 0:
        slon = "W%03d" % -ll_lon
    else:
        slon = "W%03d" % ll_lon
    return slat + slon

class SRTM():
    def __init__(self, lat, lon, dict_path):
        self.lat, self.lon = lla_ll_corner(lat, lon)
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
            print file.retrieve(url, download_file)
            return True
        else:
            print "Notice: requested srtm that is outside catalog"
            return False
        
    def parse(self):
        tilename = make_tile_name(self.lat, self.lon)
        cache_file = self.srtm_cache_dir + '/' + tilename + '.hgt.zip'
        if not os.path.exists(cache_file):
            if not self.download_srtm(tilename):
                return False
        print "Notice: parsing SRTM file:", cache_file
        #f = open(cache_file, "rb")
        zip = zipfile.ZipFile(cache_file)
        f = zip.open(tilename + '.hgt', 'r')
        contents = f.read()
        f.close()
        # read 1,442,401 (1201x1201) high-endian
        # signed 16-bit words into self.z
        self.srtm_z = struct.unpack(">1442401H", contents)
        return True
    
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
        x = np.linspace(self.lon, self.lon+1, 1201)
        y = np.linspace(self.lat, self.lat+1, 1201)
        #print x
        #print y
        self.lla_interp = scipy.interpolate.RegularGridInterpolator((x, y), srtm_pts, bounds_error=False, fill_value=-32768)
        #print self.lla_interp([-93.14530573529404, 45.220697421008396])
        #for i in range(20):
        #    x = -94.0 + random.random()
        #    y = 45 + random.random()
        #    z = self.lla_interp([x,y])
        #    print [x, y, z[0]]
        
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

# Build a gridded elevation interpolation table centered at lla_ref
# with width and height.  This is a little bit of quick feet dancing,
# but allows areas to span corners or edges of srtm tiles and attempts
# to stay on a fast path of regular grids, even though a regularly lla
# grid != a regular ned grid.
class NEDGround():
    def __init__(self, lla_ref, width_m, height_m, step_m):
        self.tile_dict = {}
        self.load_tiles(lla_ref, width_m, height_m)
        self.make_interpolator(lla_ref, width_m, height_m, step_m)
        
    def load_tiles(self, lla_ref, width_m, height_m):
        print "Notice: loading DEM tiles"
        ll_ned = np.array([[-height_m*0.5, -width_m*0.5, 0.0]])
        ur_ned = np.array([[height_m*0.5, width_m*0.5, 0.0]])
        ll_lla = navpy.ned2lla(ll_ned, lla_ref[0], lla_ref[1], lla_ref[2])
        ur_lla = navpy.ned2lla(ur_ned, lla_ref[0], lla_ref[1], lla_ref[2])
        #print ll_lla
        #print ur_lla
        lat1, lon1 = lla_ll_corner( ll_lla[0], ll_lla[1] )
        lat2, lon2 = lla_ll_corner( ur_lla[0], ur_lla[1] )
        for lat in range(lat1, lat2+1):
            for lon in range(lon1, lon2+1):
                srtm = SRTM(lat, lon, '../srtm')
                if srtm.parse():
                    srtm.make_lla_interpolator()
                    #srtm.plot_raw()
                    tile_name = make_tile_name(lat, lon)
                    self.tile_dict[tile_name] = srtm
                
    def make_interpolator(self, lla_ref, width_m, height_m, step_m):
        print "Notice: constructing NED area interpolator"
        rows = (height_m / step_m) + 1
        cols = (width_m / step_m) + 1
        #ned_pts = np.zeros((cols, rows))
        #for r in range(0,rows):
        #    for c in range(0,cols):
        #        idx = (cols*r)+c
        #        #va = self.srtm_z[idx]
        #        #if va == 65535 or va < 0 or va > 10000:
        #        #    va = 0.0
        #        #z = va
        #        ned_pts[c,r] = 0.0
        
        # build regularly gridded x,y coordinate list and ned_pts array
        n_list = np.linspace(-height_m*0.5, height_m*0.5, rows)
        e_list = np.linspace(-width_m*0.5, width_m*0.5, cols)
        #print "e's:", e_list
        #print "n's:", n_list
        ned_pts = []
        for e in e_list:
            for n in n_list:
                ned_pts.append( [n, e, 0] )

        # convert ned_pts list to lla coordinates (so it's not
        # necessarily an exact grid anymore, but we can now
        # interpolate elevations out of the lla interpolators for each
        # tile.
        navpy_pts = navpy.ned2lla(ned_pts, lla_ref[0], lla_ref[1], lla_ref[2])
        #print navpy_pts
        
        # build list of (lat, lon) points for doing actual lla
        # elevation lookup
        ll_pts = []
        for i in range( len(navpy_pts[0]) ):
            lat = navpy_pts[0][i]
            lon = navpy_pts[1][i]
            ll_pts.append( [ lon, lat ] )
        #print "ll_pts:", ll_pts

        # set all the elevations in the ned_ds list to the extreme
        # minimum value. (rows,cols) might seem funny, but (ne)d is
        # reversed from (xy)z ... don't think about it too much or
        # you'll get a headache. :-)
        ned_ds = np.zeros((rows,cols))
        ned_ds[:][:] = -32768
        #print "ned_ds:", ned_ds
        
        # for each tile loaded, interpolate as many elevation values
        # as we can, then copy the good values into ned_ds.  When we
        # finish all the loaded tiles, we should have elevations for
        # the entire range of points.
        for tile in self.tile_dict:
            zs = self.tile_dict[tile].lla_interpolate(np.array(ll_pts))
            #print zs
            # copy the good altitudes back to the corresponding ned points
            for r in range(0,rows):
                for c in range(0,cols):
                    idx = (rows*c)+r
                    if zs[idx] > -10000:
                        ned_ds[r,c] = zs[idx]

        # quick sanity check
        for r in range(0,rows):
            for c in range(0,cols):
                idx = (rows*c)+r
                if ned_ds[r,c] < -10000:
                    print "Problem interpolating elevation for:", ll_pts[idx]
                    ned_ds[r,c] = 0.0
        #print "ned_ds:", ned_ds

        # now finally build the actual grid interpolator with evenly
        # spaced ned n, e values and elevations interpolated out of
        # the srtm lla interpolator.
        self.interp = scipy.interpolate.RegularGridInterpolator((n_list, e_list), ned_ds, bounds_error=False, fill_value=-32768)

        do_plot = False
        if do_plot:
            imshow(ned_ds, interpolation='bilinear', origin='lower', cmap=cm.gray, alpha=1.0)
            grid(False)
            show()
        
        do_test = False
        if do_test:
            for i in range(40):
                ned = [(random.random()-0.5)*height_m,
                       (random.random()-0.5)*width_m,
                       0.0]
                lla = navpy.ned2lla(ned, lla_ref[0], lla_ref[1], lla_ref[2])
                #print "ned=%s, lla=%s" % (ned, lla)
                nedz = self.interp([ned[0], ned[1]])
                tile = make_tile_name(lla[0], lla[1])
                llaz = self.tile_dict[tile].lla_interpolate(np.array([lla[1], lla[0]]))
                print "nedz=%.2f llaz=%.2f" % (nedz, llaz)

    # while error > eps: find altitude at current point, new pt = proj
    # vector to current alt.
    def interpolate_vector(self, pose_dict, v):
        pose = pose_dict['ned']
        p = pose[:] # copy hopefully

        # sanity check (always assume camera pose is above ground!)
        if v[2] <= 0.0:
            return p
        
        eps = 0.01
        count = 0
        #print "start:", p
        #print "vec:", v
        #print "pose:", pose
        ground = self.interp([p[0], p[1]])
        error = abs(p[2] + ground[0])
        #print "  p=%s ground=%s error=%s" % (p, ground, error)
        while error > eps and count < 25:
            d_proj = -(pose[2] + ground[0])
            factor = d_proj / v[2]
            n_proj = v[0] * factor
            e_proj = v[1] * factor
            #print "proj = %s %s" % (n_proj, e_proj)
            p = [ pose[0] + n_proj, pose[1] + e_proj, pose[2] + d_proj ]
            #print "new p:", p
            ground = self.interp([p[0], p[1]])
            error = abs(p[2] + ground[0])
            #print "  p=%s ground=%.2f error = %.3f" % (p, ground, error)
            count += 1
        return p

    # return a list of (3d) ground intersection points for the give
    # vector list and camera pose.
    def interpolate_vectors(self, pose, v_list):
        pose_ned = pose['ned']
        pt_list = []
        for v in v_list:
            p = self.interpolate_vector(pose, v)
            pt_list.append(p)
        return pt_list
