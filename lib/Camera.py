#!/usr/bin/python

import numpy as np

from props import getNode

# camera parameters are stored in the global property tree, but this
# class provides convenient getter/setter functions

class Camera():
    def __init__(self):
        self.camera_node = getNode('/config/camera', True)

    def set_defaults(self):
        # meta data
        self.camera_node.setString('make', 'unknown')
        self.camera_node.setString('model', 'unknown')
        self.camera_node.setString('lens_model', 'unknown')

        # camera lens parameters
        self.camera_node.setFloat('focal_len_mm', 0.0)
        self.camera_node.setFloat('ccd_width_mm', 0.0)
        self.camera_node.setFloat('ccd_height_mm', 0.0)

        # camera calibration parameters
        self.camera_node.setLen('K', 9, init_val=0.0)
        self.camera_node.setLen('dist_coeffs', 5, init_val=0.0)

        # full size of camera image (these values may be needed for
        # sentera images processed through their rolling shutter
        # corrector that are not full width/height.
        self.camera_node.setFloat('width_px', 0)
        self.camera_node.setFloat('height_px', 0)

        # camera mount parameters: these are offsets from the aircraft body
        # mount_node = self.camera_node.getChild('mount', create=True)
        # mount_node.setFloat('yaw_deg', 0.0)
        # mount_node.setFloat('pitch_deg', 0.0)
        # mount_node.setFloat('roll_deg', 0.0)

    def set_meta(self, make, model, lens_model):
        self.camera_node.setString('make', make)
        self.camera_node.setString('model', model)
        self.camera_node.setString('lens_model', lens_model)

    def set_lens_params(self, ccd_width_mm, ccd_height_mm, focal_len_mm):
        self.camera_node.setFloat('ccd_width_mm', ccd_width_mm)
        self.camera_node.setFloat('ccd_height_mm', ccd_height_mm)
        self.camera_node.setFloat('focal_len_mm', focal_len_mm)
        
    def get_lens_params(self):
        return ( self.camera_node.getFloat('ccd_width_mm'), 
                 self.camera_node.getFloat('ccd_height_mm'),
                 self.camera_node.getFloat('focal_len_mm') )

    def get_K(self, optimized=False):
        """
        Form the camera calibration matrix K using 5 parameters of 
        Finite Projective Camera model.  (Note skew parameter is 0)

        See Eqn (6.10) in:
        R.I. Hartley & A. Zisserman, Multiview Geometry in Computer Vision,
        Cambridge University Press, 2004.
        """
        tmp = []
        if optimized and self.camera_node.hasChild('K_opt'):
            for i in range(9):
                tmp.append( self.camera_node.getFloatEnum('K_opt', i) )
        else:
            for i in range(9):
                tmp.append( self.camera_node.getFloatEnum('K', i) )
        K = np.copy(np.array(tmp)).reshape(3,3)
        return K
        
    def set_K(self, fx, fy, cu, cv, optimized=False):
        K = np.identity(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cu
        K[1,2] = cv
        # store as linear python list
        tmp = K.ravel().tolist()
        if optimized:
            self.camera_node.setLen('K_opt', 9)
            for i in range(9):
                self.camera_node.setFloatEnum('K_opt', i, tmp[i])
        else:
            self.camera_node.setLen('K', 9)
            for i in range(9):
                self.camera_node.setFloatEnum('K', i, tmp[i])

    # dist_coeffs = array[5] = k1, k2, p1, p2, k3
    def get_dist_coeffs(self, optimized=False):
        tmp = []
        if optimized and self.camera_node.hasChild('dist_coeffs_opt'):
            for i in range(5):
                tmp.append( self.camera_node.getFloatEnum('dist_coeffs_opt', i) )
        else:
            for i in range(5):
                tmp.append( self.camera_node.getFloatEnum('dist_coeffs', i) )
        return np.array(tmp)
    
    def set_dist_coeffs(self, dist_coeffs, optimized=False):
        if optimized:
            self.camera_node.setLen('dist_coeffs_opt', 5)
            for i in range(5):
                self.camera_node.setFloatEnum('dist_coeffs_opt', i, dist_coeffs[i])
        else:
            self.camera_node.setLen('dist_coeffs', 5)
            for i in range(5):
                self.camera_node.setFloatEnum('dist_coeffs', i, dist_coeffs[i])
        
    def set_image_params(self, width_px, height_px):
        self.camera_node.setInt('width_px', width_px)
        self.camera_node.setInt('height_px', height_px)
        
    def get_image_params(self):
        return ( self.camera_node.getInt('width_px'),
                 self.camera_node.getInt('height_px') )

    def set_mount_params(self, yaw_deg, pitch_deg, roll_deg):
        mount_node = self.camera_node.getChild('mount', True)
        mount_node.setFloat('yaw_deg', yaw_deg)
        mount_node.setFloat('pitch_deg', pitch_deg)
        mount_node.setFloat('roll_deg', roll_deg)
        self.camera_node.pretty_print()
       
    def get_mount_params(self):
        mount_node = self.camera_node.getChild('mount', True)
        return [ mount_node.getFloat('yaw_deg'),
                 mount_node.getFloat('pitch_deg'),
                 mount_node.getFloat('roll_deg') ]

    # def derive_other_params(self):
    #     K = self.get_K()
    #     fx = K[0,0]
    #     fy = K[1,1]
    #     cu = K[0,2]
    #     cv = K[1,2]
    #     width_px = self.camera_node.getFloat('width_px')
    #     height_px = self.camera_node.getFloat('height_px')
    #     ccd_width_mm = self.camera_node.getFloat('ccd_width_mm')
    #     ccd_height_mm = self.camera_node.getFloat('ccd_height_mm')
    #     focal_len_mm = self.camera_node.getFloat('focal_len_mm')
    #     if cu < 1.0 and width_px > 0:
    #         cu = width_px * 0.5
    #     if cv < 1.0 and height_px > 0:
    #         cv = height_px * 0.5
    #     if fx < 1 and focal_len_mm > 0 and width_px > 0 and ccd_width_mm > 0:
    #         fx = (focal_len_mm * width_px) / ccd_width_mm
    #     if fy < 1 and focal_len_mm > 0 and height_px > 0 and ccd_height_mm > 0:
    #         fy = (focal_len_mm * height_px) / ccd_height_mm
