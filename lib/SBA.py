#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

import sys
sys.path.insert(0, "/usr/local/opencv-2.4.11/lib/python2.7/site-packages/")

import numpy as np
import re
import subprocess

sys.path.append('../lib')
import transformations


# This is a python class that wraps the SBA package as an external
# program.
#
# The required external SBA package can be found here:
#
#     http://users.ics.forth.gr/~lourakis/sba/index.html
#
# Camera poses are given as [ R | t ] where R is the same R we use
# throughout and t is the 'tvec' as used within the cv2 system.

class SBA():
    def __init__(self, root):
        self.program = 'eucsbademo'
        self.root = root

    # write the camera (motion) parameters, feature (structure)
    # parameters, and calibration (K) to files in the project
    # directory.
    def prepair_data(self, image_list, matches_dict, K):
        # iterate through the image list and build the camera pose dictionary
        # (and a simple list of camera locations for plotting)
        f = open( self.root + '/sba-cams.txt', 'w' )
        for image in image_list:
            # try #1
            body2cam = image.get_body2cam()
            ned2body = image.get_ned2body()
            Rtotal = body2cam.dot( ned2body )
            q = transformations.quaternion_from_matrix(Rtotal)
            rvec, tvec = image.get_proj()
            s = "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (q[0], q[1], q[2], q[3],
                                                          tvec[0,0], tvec[1,0], tvec[2,0])
            f.write(s)
        f.close()

        # iterate through the matches dictionary to produce a list of matches
        f = open( self.root + '/sba-points.txt', 'w' )
        for key in matches_dict:
            feat = matches_dict[key]
            ned = np.array(feat['ned']) / 1.0
            s = "%.4f %.4f %.4f " % (ned[0], ned[1], ned[2])
            f.write(s)
            pts = feat['pts']
            s = "%d " % (len(pts))
            f.write(s)
            for p in pts:
                image_num = p[0]
                kp = image_list[image_num].kp_list[p[1]]
                s = "%d %.2f %.2f " % (image_num, kp.pt[0], kp.pt[1])
                f.write(s)
            f.write('\n')
        f.close()

        # generate the calibration matrix (K) file
        f = open( self.root + '/sba-calib.txt', 'w' )
        s = "%.4f %.4f %.4f\n" % (K[0,0], K[0,1], K[0,2])
        f.write(s)
        s = "%.4f %.4f %.4f\n" % (K[1,0], K[1,1], K[1,2])
        f.write(s)
        s = "%.4f %.4f %.4f\n" % (K[2,0], K[2,1], K[2,2])
        f.write(s)

    def run(self):
        command = []
        command.append( self.program )
        command.append( self.root + '/sba-cams.txt' )
        command.append( self.root + '/sba-points.txt' )
        command.append( self.root + '/sba-calib.txt' )
        print "Running:", command
        result = subprocess.check_output( command )

        state = ''
        mre_start = 0.0         # mre = mean reprojection error
        mre_final = 0.0         # mre = mean reprojection error
        iterations = 0
        time_msec = 0.0
        cameras = []
        features = []
        
        for line in result.split('\n'):
            if re.search('mean reprojection error', line):
                value = float(re.sub('mean reprojection error', '', line))
                if mre_start == 0.0:
                    mre_start = value
                else:
                    mre_final = value
            elif re.search('iterations=', line):
                iterations = int(re.sub('iterations=', '', line))
            elif re.search('Elapsed time:', line):
                tokens = line.split()
                time_msec = float(tokens[4])
            elif re.search('Motion parameters:', line):
                state = 'motion'
            elif re.search('Structure parameters:', line):
                state = 'structure'
            else:
                tokens = line.split()
                if state == 'motion' and len(tokens) == 7:
                    # print "camera:", np.array(tokens, dtype=float)
                    cameras.append( np.array(tokens, dtype=float) )
                elif state == 'structure' and len(tokens) == 3:
                    # print "feature:", np.array(tokens, dtype=float)
                    features.append( np.array(tokens, dtype=float) )
                # else:
                    # print "debug =", line
            
        print "Starting mean reprojection error:", mre_start
        print "Final mean reprojection error:", mre_final
        print "Iterations =", iterations
        print "Elapsed time = %.2f sec (%.2f msec)" % (time_msec/1000,
                                                       time_msec)
        return cameras, features
