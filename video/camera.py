import cv2
from math import atan2, cos, pi, sin
import numpy as np
import os

from props import PropertyNode
import props_json

import sys
sys.path.append('../scripts')
from lib import transformations

# helpful constants
d2r = pi / 180.0
r2d = 180.0 / pi

# these are fixed tranforms between ned and camera reference systems
proj2ned = np.array( [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                     dtype=float )
ned2proj = np.linalg.inv(proj2ned)

class VirtualCamera:
    config = PropertyNode()
    K = None
    IK = None
    dist = None
    PROJ = None

    def __init__(self):
        pass

    def load(self, camera_config, local_config, scale=1.0):
        if camera_config is None:
            if os.path.exists(local_config):
                # load local config file if it exists
                result = props_json.load(local_config, self.config)
                if not result:
                    print("Cannot continue with invalid camera file.")
                    quit()
            else:
                print("no camera config specified and no local camera config file found:", local_config)
                quit()
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

    def get_IK(self):
        if self.IK is None:
            self.IK = np.linalg.inv( self.get_K() )
        return self.IK

    def get_dist(self):
        if self.dist is None:
            self.dist = []
            for i in range(5):
                self.dist.append( self.config.getFloatEnum("dist_coeffs", i) )
        return self.dist

    def get_shape(self):
        return self.config.getFloat("width_px"), self.config.getFloat("height_px")

    def get_ypr(self):
        cam_yaw = self.config.getFloatEnum('mount_ypr', 0)
        cam_pitch = self.config.getFloatEnum('mount_ypr', 1)
        cam_roll = self.config.getFloatEnum('mount_ypr', 2)
        return cam_yaw, cam_pitch, cam_roll

    def set_ypr(self, cam_yaw, cam_pitch, cam_roll):
        self.config.setFloatEnum('mount_ypr', 0, cam_yaw)
        self.config.setFloatEnum('mount_ypr', 1, cam_pitch)
        self.config.setFloatEnum('mount_ypr', 2, cam_roll)

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


    # utility functions to support various efforts

    # precompute to save time
    horiz_ned = [0, 0, 0]  # any value works here (as long as it's consistent
    horiz_divs = 10
    horiz_pts = []
    for i in range(horiz_divs + 1):
        a = (float(i) * 360/float(horiz_divs)) * d2r
        n = cos(a) + horiz_ned[0]
        e = sin(a) + horiz_ned[1]
        d = 0.0 + horiz_ned[2]
        horiz_pts.append( [n, e, d] )

    # returns roll, pitch of horizon given current camera attitude.
    # Note camera attitude typically includes aircraft body attitude
    # and camera mount offset.  (The attiude values and offsets should
    # be set/updated by the calling function)
    def find_horizon(self):
        answers = []
        K = self.get_K()
        IK = self.get_IK()
        cu = K[0,2]
        cv = K[1,2]
        for i in range(self.horiz_divs):
            uv1 = self.project_ned( self.horiz_pts[i] )
            uv2 = self.project_ned( self.horiz_pts[i+1] )
            if uv1 != None and uv2 != None:
                #print(" ", uv1, uv2)
                roll, pitch = self.get_projected_attitude(uv1, uv2, IK, cu, cv)
                answers.append( (roll, pitch) )
        if len(answers) > 0:
            index = int(len(answers) / 2)
            return answers[index]
        else:
            return None, None

    # a, b are line end points, p is some other point
    # returns the closest point on ab to p (orthogonal projection)
    def ClosestPointOnLine(self, a, b, p):
        ap = p - a
        ab = b - a
        return a + np.dot(ap,ab) / np.dot(ab,ab) * ab

    # get the roll/pitch of camera orientation relative to specified
    # horizon line
    def get_projected_attitude(self, uv1, uv2, IK, cu, cv):
        # print('line:', line)
        du = uv2[0] - uv1[0]
        dv = uv1[1] - uv2[1]        # account for (0,0) at top left corner in image space
        roll = atan2(dv, du)

        if False:
            # temp test
            w = cu * 2
            h = cv * 2
            for p in [ (0, 0, 1), (w, 0, 1), (0, h, 1), (w, h, 1), (cu, cv, 1) ]:
                uvh = np.array(p)
                proj = IK.dot(uvh)
                print(p, "->", proj)

        p0 = self.ClosestPointOnLine(np.array(uv1), np.array(uv2),
                                     np.array([cu,cv]))
        uvh = np.array([p0[0], p0[1], 1.0])
        proj = IK.dot(uvh)
        #print("proj:", proj, proj/np.linalg.norm(proj))
        dot_product = np.dot(np.array([0,0,1]), proj/np.linalg.norm(proj))
        pitch = np.arccos(dot_product)
        if p0[1] < cv:
            pitch = -pitch
        #print("roll: %.1f pitch: %.1f" % (roll, pitch))
        return roll, pitch

