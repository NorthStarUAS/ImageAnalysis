#!/usr/bin/python

# write out the data in a form useful to pass to the sba (demo) program

# it appears camera poses are basically given as [ R | t ] where R is
# the same R we use throughout and t is the 'tvec'

# todo, run sba and automatically parse output ...

import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/")

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
        self.image_map_fwd = {}
        self.image_map_rev = {}

    # write the camera (motion) parameters, feature (structure)
    # parameters, and calibration (K) to files in the project
    # directory.
    def prepair_data(self, image_list, placed_images, matches_list, K,
                     use_sba=False):
        if placed_images == None:
            placed_images = set()
            # if no placed images specified, mark them all as placed
            for i in range(len(image_list)):
                placed_images.add(i)
                
        # construct the image index remapping
        self.image_map_fwd = {}
        self.image_map_rev = {}
        for i, index in enumerate(placed_images):
            self.image_map_fwd[i] = index
            self.image_map_rev[index] = i
        print self.image_map_fwd
        print self.image_map_rev
            
        # iterate through the image list and build the camera pose dictionary
        # (and a simple list of camera locations for plotting)
        f = open( self.root + '/sba-cams.txt', 'w' )
        for i, index in enumerate(placed_images):
            image = image_list[index]
            body2cam = image.get_body2cam()
            if use_sba:
                ned2body = image.get_ned2body_sba()
            else:
                ned2body = image.get_ned2body()
            Rtotal = body2cam.dot( ned2body )
            q = transformations.quaternion_from_matrix(Rtotal)
            if use_sba:
                rvec, tvec = image.get_proj_sba()
            else:
                rvec, tvec = image.get_proj()
            s = "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % (q[0], q[1], q[2], q[3],
                                                          tvec[0,0], tvec[1,0], tvec[2,0])
            f.write(s)
        f.close()

        # produce a cams file with variable K
        f = open( self.root + '/sba-cams-varK.txt', 'w' )
        for i, index in enumerate(placed_images):
            image = image_list[index]
            body2cam = image.get_body2cam()
            if use_sba:
                ned2body = image.get_ned2body_sba()
            else:
                ned2body = image.get_ned2body()
            Rtotal = body2cam.dot( ned2body )
            q = transformations.quaternion_from_matrix(Rtotal)
            if use_sba:
                rvec, tvec = image.get_proj_sba()
            else:
                rvec, tvec = image.get_proj()
            s = "%.5f %.5f %.5f %.5f %.1f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % \
                (K[0,0], K[0,2], K[1,2], K[1,1]/K[0,0], 0.0,
                 q[0], q[1], q[2], q[3],
                 tvec[0,0], tvec[1,0], tvec[2,0])
            f.write(s)
        f.close()

        # produce a cams file with variable K and D
        f = open( self.root + '/sba-cams-varKD.txt', 'w' )
        for i, index in enumerate(placed_images):
            image = image_list[index]
            body2cam = image.get_body2cam()
            if use_sba:
                ned2body = image.get_ned2body_sba()
            else:
                ned2body = image.get_ned2body()
            Rtotal = body2cam.dot( ned2body )
            q = transformations.quaternion_from_matrix(Rtotal)
            if use_sba:
                rvec, tvec = image.get_proj_sba()
            else:
                rvec, tvec = image.get_proj()
            s = "%.5f %.5f %.5f %.5f %.1f %.2f %.2f %.2f %.2f %.2f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" % \
                (K[0,0], K[0,2], K[1,2], K[1,1]/K[0,0], 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0,
                 q[0], q[1], q[2], q[3],
                 tvec[0,0], tvec[1,0], tvec[2,0])
            f.write(s)
        f.close()

        # iterate through the matches dictionary to produce a list of matches
        f = open( self.root + '/sba-points.txt', 'w' )
        for i, match in enumerate(matches_list):
            ned = np.array(match[0])
            #print type(ned), ned.size, ned
            count = 0
            for p in match[1:]:
                if p[0] in placed_images:
                    count += 1
            if ned.size == 3 and count >= 2:
                s = "%.4f %.4f %.4f  " % (ned[0], ned[1], ned[2])
                f.write(s)
                s = "%d  " % (count)
                f.write(s)
                for p in match[1:]:
                    if p[0] in placed_images:
                        local_index = self.image_map_rev[p[0]]
                        # kp = image_list[p[0]].kp_list[p[1]].pt # distorted
                        kp = image_list[p[0]].uv_list[p[1]]      # undistorted
                        s = "%d %.2f %.2f " % (local_index, kp[0], kp[1])
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

    def run_live(self, mode=''):
        command = []
        command.append( self.program )
        if mode == '':
            command.append( self.root + '/sba-cams.txt' )
            command.append( self.root + '/sba-points.txt' )
            command.append( self.root + '/sba-calib.txt' )
        elif mode == 'varK':
            command.append( self.root + '/sba-cams-varK.txt' )
            command.append( self.root + '/sba-points.txt' )
        elif mode == 'varKD':
            command.append( self.root + '/sba-cams-varKD.txt' )
            command.append( self.root + '/sba-points.txt' )
        print "Running:", command

        #result = subprocess.check_output( command )
        # bufsize=1 is line buffered
        process = subprocess.Popen( command, stdout=subprocess.PIPE)

        state = ''
        mre_start = 0.0         # mre = mean reprojection error
        mre_final = 0.0         # mre = mean reprojection error
        iterations = 0
        time_msec = 0.0
        cameras = []
        features = []

        result = process.stdout.readline()
        print result
        while result:
            for line in result.split('\n'):
                #print "line: ", line
                if re.search('mean reprojection error', line):
                    print line
                    value = float(re.sub('mean reprojection error', '', line))
                    if mre_start == 0.0:
                        mre_start = value
                    else:
                        mre_final = value
                elif re.search('damping term', line):
                    print  line 
                elif re.search('iterations=', line):
                    print line
                    iterations = int(re.sub('iterations=', '', line))
                elif re.search('Elapsed time:', line):
                    print line
                    tokens = line.split()
                    time_msec = float(tokens[4])
                elif re.search('Motion parameters:', line):
                    state = 'motion'
                elif re.search('Structure parameters:', line):
                    state = 'structure'
                else:
                    tokens = line.split()
                    if state == 'motion' and len(tokens) > 0:
                        # print "camera:", np.array(tokens, dtype=float)
                        cameras.append( np.array(tokens, dtype=float) )
                    elif state == 'structure' and len(tokens) == 3:
                        # print "feature:", np.array(tokens, dtype=float)
                        features.append( np.array(tokens, dtype=float) )
                    elif len(line):
                        print line
            # read next line
            result = process.stdout.readline()
            
        print "Starting mean reprojection error:", mre_start
        print "Final mean reprojection error:", mre_final
        print "Iterations =", iterations
        print "Elapsed time = %.2f sec (%.2f msec)" % (time_msec/1000,
                                                       time_msec)
        return cameras, features, self.image_map_fwd
