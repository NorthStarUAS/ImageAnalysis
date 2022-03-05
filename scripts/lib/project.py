"""Manage an ImageAnalysis project."""

import fnmatch
import os.path

import cv2
import numpy as np
import scipy.interpolate
from props import getNode, props_json
from tqdm import tqdm

# from . import Render
from . import camera, image, logger, state, transformations
from .logger import log


class ProjectMgr:
    """Manage a project."""

    def __init__(self, project_dir, create=False):
        """Initialize a project."""
        self.project_dir = project_dir
        self.analysis_dir = os.path.join(self.project_dir, "ImageAnalysis")
        self.image_list = []
        self.matcher_params = {
            "matcher": "FLANN",  # { FLANN or 'BF' }
            "match-ratio": 0.75,
            "filter": "fundamental",
            "image-fuzz": 40,
            "feature-fuzz": 20,
        }

        # the following member variables need to be reviewed/organized
        self.ac3d_steps = 8
        # self.render = Render.Render()
        self.dir_node = getNode("/config/directories", True)
        self.load(create)

    def set_defaults(self):
        """Set project camera defaults."""
        camera.set_defaults()  # camera defaults

    # project_dir is a new folder for all derived files
    def validate_project_dir(self, create_if_needed):
        """Validate a new folder for all derived files."""
        if not os.path.exists(self.analysis_dir):
            if create_if_needed:
                log("Creating analysis directory:", self.analysis_dir)
                os.makedirs(self.analysis_dir)
            else:
                log("Error: analysis dir doesn't exist: ", self.analysis_dir)
                return False

        # log directory
        logger.init(self.analysis_dir)

        # and make other children directories
        meta_dir = os.path.join(self.analysis_dir, "meta")
        if not os.path.exists(meta_dir):
            if create_if_needed:
                log("project: creating meta directory:", meta_dir)
                os.makedirs(meta_dir)
            else:
                log("Error: meta dir doesn't exist:", meta_dir)
                return False
        cache_dir = os.path.join(self.analysis_dir, "cache")
        if not os.path.exists(cache_dir):
            if create_if_needed:
                log("project: creating cache directory:", cache_dir)
                os.makedirs(cache_dir)
            else:
                log("Notice: cache dir doesn't exist:", cache_dir)
        state_dir = os.path.join(self.analysis_dir, "state")
        if not os.path.exists(state_dir):
            if create_if_needed:
                log("project: creating state directory:", state_dir)
                os.makedirs(state_dir)
            else:
                log("Notice: state dir doesn't exist:", state_dir)
        state.init(state_dir)

        # all is good
        return True

    def save(self):
        """Create a project dictionary and write it out as json."""
        if not os.path.exists(self.analysis_dir):
            raise FileNotFoundError("Analysis dir missing!")

        project_file = os.path.join(self.analysis_dir, "config.json")
        config_node = getNode("/config", True)
        props_json.save(project_file, config_node)

    def load(self, create=True):
        """Load a project."""
        if not self.validate_project_dir(create):
            raise FileNotFoundError("Analysis dir missing!")

        # load project configuration
        result = False
        project_file = os.path.join(self.analysis_dir, "config.json")
        config_node = getNode("/config", True)
        if os.path.isfile(project_file):
            if props_json.load(project_file, config_node):
                # fixme:
                # if 'matcher' in project_dict:
                #     self.matcher_params = project_dict['matcher']
                # root.pretty_print()
                result = True
            else:
                log("Notice: unable to load: ", project_file)
        else:
            log("project: project configuration doesn't exist:", project_file)
        if not result and create:
            log("Continuing with an empty project configuration")
            self.set_defaults()
        elif not result:
            log("Project load failed, aborting...")
            raise Exception("Can't Load Project")

        # overwrite project_dir with current location (this will get
        # saved out into the config.json, but projects could relocate
        # and it's more important to have the actual current location)
        self.dir_node.setString("project_dir", self.project_dir)

        # root.pretty_print()

    def detect_camera(self):
        """Detect a camera."""
        from . import exif  # only import if we call this fucntion

        image_dir = self.project_dir
        for file in os.listdir(image_dir):
            if (
                fnmatch.fnmatch(file, "*.jpg")
                or fnmatch.fnmatch(file, "*.JPG")
                or fnmatch.fnmatch(file, "*.tif")
                or fnmatch.fnmatch(file, "*.TIF")
            ):
                image_file = os.path.join(image_dir, file)
                cam, make, model, lens = exif.get_camera_info(image_file)
                break
        return cam, make, model, lens

    def load_images_info(self):
        """Load a list of image information."""
        # wipe image list (so we don't double load)
        self.image_list = []

        # load image meta info
        meta_dir = os.path.join(self.analysis_dir, "meta")
        images_node = getNode("/images", True)
        for file in sorted(os.listdir(meta_dir)):
            if fnmatch.fnmatch(file, "*.json"):
                name, ext = os.path.splitext(file)
                image_node = images_node.getChild(name, True)
                props_json.load(os.path.join(meta_dir, file), image_node)
                i1 = image.Image(self.analysis_dir, name)
                self.image_list.append(i1)

    def load_features(self, descriptors=False):
        """Load feature keypoints and descriptors."""
        if descriptors:
            log("Loading feature keypoints and descriptors:")
        else:
            log("Loading feature keypoints:")
        for im in tqdm(self.image_list, smoothing=0.05):
            im.load_features()
            if descriptors:
                im.load_descriptors()

    def load_match_pairs(self, extra_verbose=False):
        """Load the match pairs."""
        log("Loading keypoint (pair) matches:")
        for im in tqdm(self.image_list, smoothing=0.05):
            im.load_matches()
            wipe_list = []
            for name in im.match_list:
                if self.find_image_by_name(name) is None:
                    print(im.name, "references", name, "which does not exist")
                    wipe_list.append(name)
            for name in wipe_list:
                del im.match_list[name]

    def save_images_info(self):
        """Create a project dictionary and write it out as json."""
        if not os.path.exists(self.analysis_dir):
            print("Error: project doesn't exist:", self.analysis_dir)
            return

        meta_dir = os.path.join(self.analysis_dir, "meta")
        images_node = getNode("/images", True)
        for name in images_node.getChildren():
            image_node = images_node.getChild(name, True)
            image_path = os.path.join(meta_dir, name + ".json")
            props_json.save(image_path, image_node)

    def set_matcher_params(self, mparams):
        """Set matcher parameters."""
        self.matcher_params = mparams

    def show_features_image(self, image):
        """Show image features."""
        return image.show_features()

    def show_features_images(self, name=None):
        """Show feature images."""
        for im in self.image_list:
            result = self.show_features_image(im)
            if result == 27 or result == ord("q"):
                break

    def find_image_by_name(self, name):
        """Find image by name."""
        for i in self.image_list:
            if i.name == name:
                return i
        return None

    def find_index_by_name(self, name):
        """Find index by name."""
        for i, img in enumerate(self.image_list):
            if img.name == name:
                return i
        return None

    def compute_ned_reference_lla(self):
        """Compute a center reference location (lon, lat) for the group."""
        # requires images to have their location computed/loaded
        lon_sum = 0.0
        lat_sum = 0.0
        count = 0
        images_node = getNode("/images", True)
        for name in images_node.getChildren():
            image_node = images_node.getChild(name, True)
            pose_node = image_node.getChild("aircraft_pose", True)
            if pose_node.hasChild("lon_deg") and pose_node.hasChild("lat_deg"):
                lon_sum += pose_node.getFloat("lon_deg")
                lat_sum += pose_node.getFloat("lat_deg")
                count += 1
        ned_node = getNode("/config/ned_reference", True)
        ned_node.setFloat("lat_deg", lat_sum / count)
        ned_node.setFloat("lon_deg", lon_sum / count)
        ned_node.setFloat("alt_m", 0.0)

    def undistort_uvlist(self, image, uv_orig):
        """Undistort the (u,v) list by camera internals."""
        if len(uv_orig) == 0:
            return []
        # camera parameters
        dist_coeffs = np.array(camera.get_dist_coeffs())
        k = camera.get_k()  # noqa: N806
        # assemble the points in the proper format
        uv_raw = np.zeros((len(uv_orig), 1, 2), dtype=np.float32)
        for i, kp in enumerate(uv_orig):
            uv_raw[i][0] = (kp[0], kp[1])
        # do the actual undistort
        uv_new = cv2.undistortPoints(uv_raw, k, dist_coeffs, P=k)
        # return the results in an easier format
        result = []
        for i, uv in enumerate(uv_new):
            result.append(uv_new[i][0])
            # print "  orig = %s  undistort = %s" % (uv_raw[i][0], uv_new[i][0]
        return result

    def undistort_image_keypoints(self, image, optimized=False):
        """Undidistort pixel location of keypoints, by camera internals."""
        if len(image.kp_list) == 0:
            return
        k = camera.get_k(optimized)
        uv_raw = np.zeros((len(image.kp_list), 1, 2), dtype=np.float32)
        for i, kp in enumerate(image.kp_list):
            uv_raw[i][0] = (kp.pt[0], kp.pt[1])
        dist_coeffs = camera.get_dist_coeffs(optimized)
        uv_new = cv2.undistortPoints(uv_raw, k, np.array(dist_coeffs), P=k)
        image.uv_list = []
        for i, uv in enumerate(uv_new):
            image.uv_list.append(uv_new[i][0])
            # print("  orig = %s  undistort = %s" % (uv_raw[i][0], uv_new[i]

    # for each feature in each image, compute the undistorted pixel
    # location (from the calibrated distortion parameters)
    def undistort_keypoints(self, optimized=False):
        """Undistort all keypoints in the project."""
        log("Undistorting keypoints:")
        for im in tqdm(self.image_list):
            self.undistort_image_keypoints(im, optimized)

    # for each uv in the provided uv list, apply the distortion
    # formula to compute the original distorted value.
    def redistort(self, uv_list, optimized=False):
        """Distort each UV to find original pixel values."""
        # TODO: revert this to the opencv function.
        k = camera.get_k(optimized)
        dist_coeffs = camera.get_dist_coeffs(optimized)
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]
        k1, k2, p1, p2, k3 = dist_coeffs

        uv_distorted = []
        for pt in uv_list:
            x = (pt[0] - cx) / fx
            y = (pt[1] - cy) / fy

            # Compute radius^2
            r2 = x ** 2 + y ** 2
            r4, r6 = r2 ** 2, r2 ** 3

            # Compute tangential distortion
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

            # Compute radial factor
            l_r = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            ud = l_r * x + dx
            vd = l_r * y + dy
            uv_distorted.append([ud * fx + cx, vd * fy + cy])

        return uv_distorted

    def compute_kp_usage(self, all_kp=False):
        """Determine feature usage in matching pairs."""
        log("[orig] Determining feature usage in matching pairs...")
        # but they may have different scaling or other attributes important
        # during feature matching
        if all_kp:
            for im in self.image_list:
                im.kp_used = np.ones(len(im.kp_list), np.bool_)
        else:
            for im in self.image_list:
                im.kp_used = np.zeros(len(im.kp_list), np.bool_)
            for i1 in self.image_list:
                # print(i1.name, len(i1.match_list))
                for key in i1.match_list:
                    matches = i1.match_list[key]
                    i2 = self.find_image_by_name(key)
                    if i2 is not None:
                        # ignore match pairs not from our area set
                        for k, pair in enumerate(matches):
                            i1.kp_used[pair[0]] = True
                            i2.kp_used[pair[1]] = True

    def compute_kp_usage_new(self, matches_direct):
        """Determine feature usage in matching pairs."""
        log("[new] Determining feature usage in matching pairs...")
        for im in self.image_list:
            im.kp_used = np.zeros(len(im.kp_list), np.bool_)
        for match in matches_direct:
            for p in match[1:]:
                im = self.image_list[p[0]]
                im.kp_used[p[1]] = True

    def project_sba(self, i_k, image):
        """Project the (u,v) pixels for the specified image.

        Using the current SBA pose and write to image.vec_list.
        """
        vec_list = []
        body2ned = image.get_body2ned_sba()
        cam2body = image.get_cam2body()
        for uv in image.uv_list:
            uvh = np.array([uv[0], uv[1], 1.0])
            proj = body2ned.dot(cam2body).dot(i_k).dot(uvh)
            proj_norm = transformations.unit_vector(proj)
            vec_list.append(proj_norm)
        return vec_list

    def fast_project_keypoints_to_3d(self, sss):
        """Project keypoints to 3d using a LUT."""
        print("Projecting keypoints to 3d:")
        k = camera.get_k()
        i_k = np.linalg.inv(k)
        for im in tqdm(self.image_list):
            # build a regular grid of uv coordinates
            w, h = im.get_size()
            steps = 32
            u_grid = np.linspace(0, w - 1, steps + 1)
            v_grid = np.linspace(0, h - 1, steps + 1)
            uv_raw = []
            for u in u_grid:
                for v in v_grid:
                    uv_raw.append([u, v])

            uv_filt = uv_raw
            # project the grid out into vectors
            body2ned = im.get_body2ned()  # IR

            # M is a transform to map the lens coordinate system (at
            # zero roll/pitch/yaw to the ned coordinate system at zero
            # roll/pitch/yaw).  It is essentially a +90 pitch followed
            # by +90 roll (or equivalently a +90 yaw followed by +90
            # pitch.)
            cam2body = im.get_cam2body()

            vec_list = project_vectors(i_k, body2ned, cam2body, uv_filt)

            # intersect the vectors with the surface to find the 3d points
            ned, ypr, quat = im.get_camera_pose()
            coord_list = sss.interpolate_vectors(ned, vec_list)

            # filter the coordinate list for bad interpolation
            for i in reversed(range(len(coord_list))):
                if np.isnan(coord_list[i][0]):
                    print("rejecting ground interpolation fault:", uv_filt[i])
                    coord_list.pop(i)
                    uv_filt.pop(i)

            # build the multidimenstional interpolator that relates
            # undistored uv coordinates to their 3d location.  Note we
            # could also relate the original raw/distored points to
            # their 3d locations and interpolate from the raw uv's,
            # but we already have a convenient list of undistored uv
            # points.
            g = scipy.interpolate.LinearNDInterpolator(uv_filt, coord_list)

            # interpolate all the keypoints now to approximate their
            # 3d locations
            im.coord_list = []
            for i, uv in enumerate(im.uv_list):
                if im.kp_used[i]:
                    coord = g(uv)
                    # coord[0] is the 3 element vector
                    if not np.isnan(coord[0][0]):
                        im.coord_list.append(coord[0])
                    else:
                        im.coord_list.append(np.zeros(3))
                else:
                    im.coord_list.append(np.zeros(3) * np.nan)

    def fast_project_keypoints_to_ground(self, ground_m, cam_dict=None):
        """Project keypoints to ground."""
        for im in tqdm(self.image_list):
            k = camera.get_k()
            i_k = np.linalg.inv(k)

            # project the grid out into vectors
            if cam_dict is None:
                body2ned = im.get_body2ned()  # IR
            else:
                body2ned = im.rvec_to_body2ned(cam_dict[im.name]["rvec"])

            # M is a transform to map the lens coordinate system (at
            # zero roll/pitch/yaw to the ned coordinate system at zero
            # roll/pitch/yaw).  It is essentially a +90 pitch followed
            # by +90 roll (or equivalently a +90 yaw followed by +90
            # pitch.)
            cam2body = im.get_cam2body()

            vec_list = project_vectors(i_k, body2ned, cam2body, im.uv_list)

            # intersect the vectors with the surface to find the 3d points
            if cam_dict is None:
                pose = im.camera_pose
            else:
                pose = cam_dict[im.name]
            im.coord_list = intersect_vectors_with_ground_plane(
                pose["ned"], ground_m, vec_list
            )


# project the list of (u, v) pixels from image space into camera
# space, remap that to a vector in ned space (for camera
# ypr=[0,0,0], and then transform that by the camera pose, returns
# the vector from the camera, through the pixel, into ned space
def project_vectors(i_k, body2ned, cam2body, uv_list):
    """Project the list of (u,v) pixels from image space to NED space."""
    proj_list = []
    for uv in uv_list:
        uvh = np.array([uv[0], uv[1], 1.0])
        proj = body2ned.dot(cam2body).dot(i_k).dot(uvh)
        proj_norm = transformations.unit_vector(proj)
        proj_list.append(proj_norm)

    return proj_list


# given a set of vectors in the ned frame, and a starting point.
# Find the ground intersection point.  For any vectors which point into
# the sky, return just the original reference/starting point.
def intersect_vectors_with_ground_plane(pose_ned, ground_m, v_list):
    """Intersect vectors with the ground plane."""
    pt_list = []
    for v in v_list:
        # solve projection
        p = pose_ned
        if v[2] > 0.0:
            d_proj = -(pose_ned[2] + ground_m)
            factor = d_proj / v[2]
            n_proj = v[0] * factor
            e_proj = v[1] * factor
            p = [p[0] + n_proj, p[1] + e_proj, p[2] + d_proj]
        pt_list.append(p)
    return pt_list
