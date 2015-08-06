#!/usr/bin/python

import json
import numpy as np
import os.path

class Camera():
    def __init__(self):
        # note: store parameters inside a dict() to make life easy for
        # loading/saving.  We don't need fast access to these values
        self.camera_dict = {}
        
        cd = self.camera_dict

        # camera lens parameters
        cd['horiz-mm'] = 0.0
        cd['vert-mm'] = 0.0
        cd['focal-len-mm'] = 0.0

        # camera calibration parameters
        cd['fx'] = 0.0
        cd['fy'] = 0.0
        cd['cu'] = 0.0
        cd['cv'] = 0.0
        cd['kcoeffs'] = [0.0]*5
        cd['skew'] = 0.0

        cd['fx-std'] = 0.0
        cd['fy-std'] = 0.0
        cd['cu-std'] = 0.0
        cd['cv-std'] = 0.0
        cd['kcoeffs-std'] = [0.0]*5
        cd['skew-std'] = 0.0

        # full size of camera image (these values may be needed for
        # sentera images processed through their rolling shutter
        # corrector that are not full width/height.
        cd['width-px'] = 0
        cd['height-px'] = 0

        # camera mount parameters: these are offsets from the aircraft body
        cd['yaw-deg'] = 0.0
        cd['pitch-deg'] = 0.0
        cd['roll-deg'] = 0.0

    def save(self, project_dir):
        # create a dictionary and write it out as json
        if not os.path.exists(project_dir):
            print "Error: project doesn't exist =", project_dir
            return
        
        camera_file = project_dir + "/Camera.json"
        try:
            f = open(camera_file, 'w')
            json.dump(self.camera_dict, f, indent=4, sort_keys=True)
            f.close()
        except IOError as e:
            print "Save camera(): I/O error({0}): {1}".format(e.errno, e.strerror)
            return
        except:
            raise

    def load(self, project_dir):
        camera_file = project_dir + "/Camera.json"
        try:
            f = open(camera_file, 'r')
            self.camera_dict = json.load(f)
            f.close()
        except:
            print "Notice: unable to read =", camera_file
            print "Continuing with an empty camera configuration"

    def set_lens_params(self, horiz_mm, vert_mm, focal_len_mm):
        self.camera_dict['horiz-mm'] = horiz_mm
        self.camera_dict['vert-mm'] = vert_mm
        self.camera_dict['focal-len-mm'] = focal_len_mm
        
    def get_lens_params(self):
        return \
            self.camera_dict['horiz-mm'], \
            self.camera_dict['vert-mm'], \
            self.camera_dict['focal-len-mm']

    # kcoeffs = array[5]
    def set_calibration_params(self, fx, fy, cu, cv, kcoeffs, skew):
        self.camera_dict['fx'] = fx
        self.camera_dict['fy'] = fy
        self.camera_dict['cu'] = cu
        self.camera_dict['cv'] = cv
        self.camera_dict['kcoeffs'] = kcoeffs 
        self.camera_dict['skew'] = skew

    def get_calibration_params(self):
        return \
            self.camera_dict['fx'], \
            self.camera_dict['fy'], \
            self.camera_dict['cu'], \
            self.camera_dict['cv'], \
            self.camera_dict['kcoeffs'], \
            self.camera_dict['skew']
        
    def set_calibration_std(self, fx_std, fy_std, cu_std, cv_std,
                            kcoeffs_std, skew_std):
        self.camera_dict['fx-std'] = fx_std
        self.camera_dict['fy-std'] = fy_std
        self.camera_dict['cu-std'] = cu_std
        self.camera_dict['cv-std'] = cv_std
        self.camera_dict['kcoeffs-std'] = kcoeffs_std
        self.camera_dict['skew-std'] = skew_std

    def get_calibration_std(self):
        return \
            self.camera_dict['fx-std'], \
            self.camera_dict['fy-std'], \
            self.camera_dict['cu-std'], \
            self.camera_dict['cv-std'], \
            self.camera_dict['kcoeffs-std'], \
            self.camera_dict['skew-std']
    
    def set_image_params(self, width_px, height_px):
        self.camera_dict['width-px'] = width_px
        self.camera_dict['height-px'] = height_px
        
    def get_image_params(self):
        return \
            self.camera_dict['width-px'], \
            self.camera_dict['height-px']

    def set_mount_params(self, yaw_deg, pitch_deg, roll_deg):
        self.camera_dict['yaw-deg'] = yaw_deg
        self.camera_dict['pitch-deg'] = pitch_deg
        self.camera_dict['roll-deg'] = roll_deg
       
    def get_mount_params(self):
        return \
            self.camera_dict['yaw-deg'], \
            self.camera_dict['pitch-deg'], \
            self.camera_dict['roll-deg']

    def get_K(self):
        """
        Form camera calibration matrix K using 5 parameters of 
        Finite Projective Camera model.

        See Eqn (6.10) in:
        R.I. Hartley & A. Zisserman, Multiview Geometry in Computer Vision,
        Cambridge University Press, 2004.
        """
        fx, fy, cu, cv, kcoeffs, skew = self.get_calibration_params()
        K = np.array([ [fx, skew, cu],
                       [ 0,   fy, cv],
                       [ 0,    0,  1] ], dtype=float)
        return K
