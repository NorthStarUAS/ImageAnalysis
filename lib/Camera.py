#!/usr/bin/python

import json
import numpy as np
import os.path

class Camera():
    def __init__(self):
        # note: store parameters inside a dict() to make life easy for
        # loading/saving.  We don't need fast access to these values
        self.camera_dict = {}
        
        # camera lens parameters
        self.camera_dict['horiz_mm'] = 0.0
        self.camera_dict['vert_mm'] = 0.0
        self.camera_dict['focal_len_mm'] = 0.0

        # camera calibration parameters
        self.camera_dict['K'] = [0.0]*9
        self.camera_dict['dist_coeffs'] = [0.0]*5

        # full size of camera image (these values may be needed for
        # sentera images processed through their rolling shutter
        # corrector that are not full width/height.
        self.camera_dict['width_px'] = 0
        self.camera_dict['height_px'] = 0

        # camera mount parameters: these are offsets from the aircraft body
        self.camera_dict['mount_ypr'] = [ 0.0, 0.0, 0.0 ]

    def save(self, project_dir):
        # create a dictionary and write it out as json
        if not os.path.exists(project_dir):
            print("Error: project doesn't exist =", project_dir)
            return
        
        camera_file = os.path.join(project_dir, "Camera.json")
        try:
            f = open(camera_file, 'w')
            json.dump(self.camera_dict, f, indent=4, sort_keys=True)
            f.close()
        except IOError as e:
            print("Save camera(): I/O error({0}): {1}".format(e.errno, e.strerror))
            return
        except:
            raise

    def load(self, project_dir):
        camera_file = os.path.join(project_dir, "Camera.json")
        try:
            f = open(camera_file, 'r')
            self.camera_dict = json.load(f)
            f.close()
        except:
            print("Notice: unable to read =", camera_file)
            print("Continuing with an empty camera configuration")

    def set_lens_params(self, horiz_mm, vert_mm, focal_len_mm):
        self.camera_dict['horiz_mm'] = horiz_mm
        self.camera_dict['vert_mm'] = vert_mm
        self.camera_dict['focal_len_mm'] = focal_len_mm
        
    def get_lens_params(self):
        return ( self.camera_dict['horiz_mm'],
                 self.camera_dict['vert_mm'],
                 self.camera_dict['focal_len_mm'] )

    def get_K(self, scale=1.0, optimized=False):
        """
        Form the camera calibration matrix K using 5 parameters of 
        Finite Projective Camera model.  (Note skew parameter is 0)

        See Eqn (6.10) in:
        R.I. Hartley & A. Zisserman, Multiview Geometry in Computer Vision,
        Cambridge University Press, 2004.
        """
        tmp = self.camera_dict['K']
        if optimized and 'K_opt' in self.camera_dict:
            tmp = self.camera_dict['K_opt']
        K = np.copy(np.array(tmp)).reshape(3,3)
        K[0,0] *= scale
        K[1,1] *= scale
        K[0,2] *= scale
        K[1,2] *= scale
        #print('stored K:', self.camera_dict['K'], 'scaled K:', K)
        return K
        
    def set_K(self, fx, fy, cu, cv, optimized=False):
        K = np.identity(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cu
        K[1,2] = cv
        # store as linear python list
        if optimized:
            self.camera_dict['K_opt'] = K.ravel().tolist()
        else:
            self.camera_dict['K'] = K.ravel().tolist()

    # dist_coeffs = array[5] = k1, k2, p1, p2, k3
    def get_dist_coeffs(self, optimized=False):
        dist_coeffs = self.camera_dict['dist_coeffs']
        if optimized and 'dist_coeffs_opt' in self.camera_dict:
            dist_ceoffs = self.camera_dict['dist_coeffs_opt']
        return dist_coeffs
    
    def set_dist_coeffs(self, dist_coeffs, optimized=False):
        if optimized:
            self.camera_dict['dist_coeffs_opt'] = dist_coeffs
        else:
            self.camera_dict['dist_coeffs'] = dist_coeffs
        
    def set_image_params(self, width_px, height_px):
        self.camera_dict['width_px'] = width_px
        self.camera_dict['height_px'] = height_px
        
    def get_image_params(self):
        return ( self.camera_dict['width_px'],
                 self.camera_dict['height_px'] )

    def set_mount_params(self, yaw_deg, pitch_deg, roll_deg):
        self.camera_dict['mount_ypr'] = [yaw_deg, pitch_deg, roll_deg]
       
    def get_mount_params(self):
        return self.camera_dict['mount_ypr']

    def derive_other_params(self):
        K = self.get_K()
        fx = K[0,0]
        fy = K[1,1]
        cu = K[0,2]
        cv = K[1,2]
        width_px = self.camera_dict['width_px']
        height_px = self.camera_dict['height_px']
        horiz_mm = self.camera_dict['horiz_mm']
        vert_mm = self.camera_dict['vert_mm']
        focal_len_mm = self.camera_dict['focal_len_mm']
        if cu < 1.0 and width_px > 0:
            cu = width_px * 0.5
        if cv < 1.0 and height_px > 0:
            cv = height_px * 0.5
        if fx < 1 and focal_len_mm > 0 and width_px > 0 and horiz_mm > 0:
            fx = (focal_len_mm * width_px) / horiz_mm
        if fy < 1 and focal_len_mm > 0 and height_px > 0 and vert_mm > 0:
            fy = (focal_len_mm * height_px) / vert_mm
