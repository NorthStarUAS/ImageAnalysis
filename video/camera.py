import numpy as np
import os

from props import PropertyNode
import props_json

class VirtualCamera:
    config = PropertyNode()
    K = None
    dist = None
    
    def __init__(self):
        pass

    def load(self, camera_config, local_config):
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
