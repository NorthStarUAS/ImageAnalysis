import cv2
import math
import numpy as np
import os

from props import PropertyNode
import props_json

import sys
sys.path.append('../scripts')
from lib import transformations

# helpful constants
d2r = math.pi / 180.0
r2d = 180.0 / math.pi

# these are fixed tranforms between ned and camera reference systems
proj2ned = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                     dtype=float )
ned2proj = np.linalg.inv(proj2ned)

class VirtualCamera:
    config = PropertyNode()
    K = None
    dist = None
    PROJ = None
    
    def __init__(self):
        pass

    def load(self, camera_config, local_config, scale=1.0):
        if camera_config is None:
            if os.path.exists(local_config):
                # load local config file if it exists
                props_json.load(local_config, self.config)
            else:
                print("no camera config specifiec and no local camera config file found:", local_config)
                return False
        else:
            # seed the camera calibration and distortion coefficients
            # from a known camera config
            print("Setting camera config from:", camera_config)
            props_json.load(camera_config, self.config)
            self.config.setString('name', camera_config)
            props_json.save(local_config, self.config)
        self.get_K()
        if scale:
            # adjust effective K to account for scaling
            self.K = self.K * scale
            self.K[2,2] = 1.0
        self.config.setLen('mount_ypr', 3, 0)

    def save(self, local_config):
        props_json.save(local_config, self.config)
        
    def get_name(self):
        return self.config.getString('name')

    def get_K(self):
        if self.K is None:
            K_list = []
            for i in range(9):
                K_list.append( self.config.getFloatEnum('K', i) )
            self.K = np.copy(np.array(K_list)).reshape(3,3)
        return self.K

    def get_dist(self):
        if self.dist is None:
            self.dist = []
            for i in range(5):
                self.dist.append( self.config.getFloatEnum("dist_coeffs", i) )
        return self.dist
    
    def get_ypr(self):
        cam_yaw = self.config.getFloatEnum('mount_ypr', 0)
        cam_pitch = self.config.getFloatEnum('mount_ypr', 1)
        cam_roll = self.config.getFloatEnum('mount_ypr', 2)
        return cam_yaw, cam_pitch, cam_roll

    def set_yaw(self, cam_yaw):
        self.config.setFloatEnum('mount_ypr', 0, cam_yaw)
        
    def set_pitch(self, cam_pitch):
        self.config.setFloatEnum('mount_ypr', 1, cam_pitch)
        
    def set_roll(self, cam_roll):
        self.config.setFloatEnum('mount_ypr', 2, cam_roll)

    def update_PROJ(self, ned, yaw_rad, pitch_rad, roll_rad):
        cam_yaw, cam_pitch, cam_roll = self.get_ypr()
        body2cam = transformations.quaternion_from_euler( cam_yaw * d2r,
                                                          cam_pitch * d2r,
                                                          cam_roll * d2r,
                                                          'rzyx')

        # this function modifies the parameters you pass in so, avoid
        # getting our data changed out from under us, by forcing copies
        # (a = b, wasn't sufficient, but a = float(b) forced a copy.
        tmp_yaw = float(yaw_rad)
        tmp_pitch = float(pitch_rad)
        tmp_roll = float(roll_rad)    
        ned2body = transformations.quaternion_from_euler(tmp_yaw,
                                                         tmp_pitch,
                                                         tmp_roll,
                                                         'rzyx')
        #body2ned = transformations.quaternion_inverse(ned2body)

        #print 'ned2body(q):', ned2body
        ned2cam_q = transformations.quaternion_multiply(ned2body, body2cam)
        ned2cam = np.matrix(transformations.quaternion_matrix(np.array(ned2cam_q))[:3,:3]).T
        #print 'ned2cam:', ned2cam
        R = ned2proj.dot( ned2cam )
        rvec, jac = cv2.Rodrigues(R)
        tvec = -np.matrix(R) * np.matrix(ned).T
        R, jac = cv2.Rodrigues(rvec)
        # is this R the same as the earlier R?
        self.PROJ = np.concatenate((R, tvec), axis=1)
        #print 'PROJ:', PROJ
        #print lat_deg, lon_deg, altitude, ref[0], ref[1], ref[2]
        #print ned
        
        return self.PROJ

    # project from ned coordinates to image uv coordinates using
    # camera K and PROJ matrices
    def project_ned(self, ned):
        uvh = self.K.dot( self.PROJ.dot( [ned[0], ned[1], ned[2], 1.0] ).T )
        if uvh[2] > 0.2:
            uvh /= uvh[2]
            uv = ( int(round(np.squeeze(uvh[0,0]))),
                   int(round(np.squeeze(uvh[1,0]))) )
            return uv
        else:
            return None

    # project from camera 3d coordinates to image uv coordinates using
    # camera K matrix
    def project_xyz(self, v):
        uvh = self.K.dot( [v[0], v[1], v[2]] )
        # print(uvh)
        if uvh[2] > 0.2:
            uvh /= uvh[2]
            uv = ( int(round(uvh[0])),
                   int(round(uvh[1])) )
            # print(uv)
            return uv
        else:
            return None

