# This is to run the string of core functions together as described in the readme as a single run to produce the end result

import os, sys, argparse
# add the current dir to the sys path to allow for use of the other scripts
sys.path.append(os.path.dirname(__file__))

import lib

from import_proxy import create_project, set_camera_config, \
        set_poses, \
        detect_features, \
        matching, clean_and_combine_matches, match_triangulation, image_groups, \
        optimize, mre_by_image, colocated_feats, \
        render_model2, delaunay5, \
        explore

def standard_project(project_dir, camera, max_camera_angle, detection_options, matching_options, match_trig_options, optmize_options, mre_options, colocated_options, render_options, delaunay_group):
        # TODO: Find every part of the script that creates a popup or requires interaction, and make into a feature flag that can be turned off for a standard run.

        # 1a-create-project.py
        print("         Creating the standard Project")
        create_project.new_project(project_dir)

        # 1b-set-camera-config.py
        print("         Setting the camera config")
        set_camera_config.set_camera(project_dir, camera)

        # 2a-set-poses.py
        print("         Setting the camera POSEs")
        set_poses.set_pose(project_dir, max_camera_angle)

        # 3a-detect-features.py
        print("         Detecting features in the images")
        detect_features.detect(project_dir, detection_options)

        # 4a-matching.py
        print("         Matching the features in the image")
        matching.match(project_dir, matching_options)

        # 4b-clean-and-combine-matches.py
        print("         Cleaning and combining the matches")
        clean_and_combine_matches.clean(project_dir)

        # 4c-match-triangulation.py
        print("         Matching and triangulation")
        match_triangulation.match_trig(project_dir, match_trig_options)

        # 4d-image-groups.py
        print("         Image Grouping")
        image_groups.group(project_dir)

        # 5a-optimize.py
        print("         Optimizing")
        optimize.optmizer(project_dir, optmize_options)

        # 5b-mre-by-image.py
        print("         MRE by Image")
        mre_by_image.mre(project_dir, mre_options)

        # 5b-colocated-feats.py
        print("         Colocate the features")
        colocated_feats.colocated(project_dir, colocated_options)

        # 6a-render-model2.py
        print("         Rendering the scene")
        render_model2.render(project_dir, render_options)

        # 6b-delaunay5.py
        print("         Delaunay 5")
        delaunay5.delaunay(project_dir, delaunay_group)

        # 7a-explore.py
        print("         Explore your direct projected project")
        explore.run(project_dir)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Create an empty project.')
        # Required parameters:
        parser.add_argument('--project', required=True, help='Directory with a set of aerial images.')
        parser.add_argument('--camera', required=True, help='camera config file path')
        parser.add_argument('--pix4d', required=True, help='use the specified pix4d csv file (lat,lon,alt,roll,pitch,yaw)')
        #
        # All other parameters are optional and will use the default provided:
        #
        parser.add_argument('--yaw-deg', type=float, default=0.0, help='camera yaw mounting offset from aircraft')
        parser.add_argument('--pitch-deg', type=float, default=-90.0, help='camera pitch mounting offset from aircraft')
        parser.add_argument('--roll-deg', type=float, default=0.0, help='camera roll mounting offset from aircraft')
        parser.add_argument('--meta', help='use the specified image-metadata.txt file (lat,lon,alt,yaw,pitch,roll)')
        parser.add_argument('--max-angle', type=float, default=25.0, help='max pitch or roll angle for image inclusion')
        parser.add_argument('--scale', type=float, default=0.5, help='scale images before detecting features, this acts much like a noise filter')
        # detection options:
        parser.add_argument('--detector', default='SIFT', choices=['SIFT', 'SURF', 'ORB', 'Star'])
        parser.add_argument('--sift-max-features', default=30000, help='maximum SIFT features')
        parser.add_argument('--surf-hessian-threshold', default=600, help='hessian threshold for surf method')
        parser.add_argument('--surf-noctaves', default=4, help='use a bigger number to detect bigger features')
        parser.add_argument('--orb-max-features', default=2000, help='maximum ORB features')
        parser.add_argument('--grid-detect', default=1, help='run detect on gridded squares for (maybe) better feature distribution, 4 is a good starting value, only affects ORB method')
        parser.add_argument('--star-max-size', default=16, help='4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128')
        parser.add_argument('--star-response-threshold', default=30)
        parser.add_argument('--star-line-threshold-projected', default=10)
        parser.add_argument('--star-line-threshold-binarized', default=8)
        parser.add_argument('--star-suppress-nonmax-size', default=5)
        parser.add_argument('--reject-margin', default=0, help='reject features within this distance of the image outer edge margin')
        parser.add_argument('--show', action='store_true', help='show features as we detect them')
        # matching options:
        parser.add_argument('--matcher', default='FLANN', choices=['FLANN', 'BF'])
        parser.add_argument('--match-ratio', default=0.75, type=float, help='match ratio')
        parser.add_argument('--min-pairs', default=25, type=int, help='minimum matches between image pairs to keep')
        parser.add_argument('--min-dist', default=0, type=float, help='minimum 2d camera distance for pair comparison')
        parser.add_argument('--max-dist', default=75, type=float, help='maximum 2d camera distance for pair comparison')
        parser.add_argument('--filter', default='gms', choices=['gms', 'homography', 'fundamental', 'essential', 'none'])
        parser.add_argument('--min-chain-length', type=int, default=3, help='minimum match chain length (3 recommended)')
        #parser.add_argument('--ground', type=float, help='ground elevation in meters')
        # match triangulation options:
        parser.add_argument('--trig-group', type=int, default=0, help='group number')
        parser.add_argument('--method', default='srtm', choices=['srtm', 'triangulate'])
        # optmize options:
        parser.add_argument('--optmize-group', type=int, default=0, help='group number')
        parser.add_argument('--refine', action='store_true', help='refine a previous optimization.')
        parser.add_argument('--cam-calibration', action='store_true', help='include camera calibration in the optimization.')
        # mre options:
        parser.add_argument('--mre-group', type=int, default=0, help='group number')
        parser.add_argument('--stddev', type=float, default=5, help='how many stddevs above the mean for auto discarding features')
        parser.add_argument('--initial-pose', action='store_true', default=False, help='work on initial pose, not optimized pose')
        parser.add_argument('--strong', action='store_true', help='remove entire match chain, not just the worst offending element.')
        parser.add_argument('--interactive', action='store_true', help='interactively review reprojection errors from worst to best and select for deletion or keep.')
        # colocated features options:
        parser.add_argument('--colocated-group', type=int, default=0, help='group index')
        parser.add_argument('--min-angle', type=float, default=1.0, help='max feature angle')
        # render options:
        parser.add_argument('--render-group', type=int, default=0, help='group index')
        parser.add_argument('--texture-resolution', type=int, default=512, help='texture resolution (should be 2**n, so numbers like 256, 512, 1024, etc.')
        parser.add_argument('--srtm', action='store_true', help='use srtm elevation')
        parser.add_argument('--ground', type=float, help='force ground elevation in meters')
        parser.add_argument('--direct', action='store_true', help='use direct pose')
        # delaunay options:
        parser.add_argument('--delaunay-group', type=int, default=0, help='group index')

        args = parser.parse_args()

        detection_options = [args.detector, args.scale, args.sift_max_features, 
                args.surf_hessian_threshold, args.surf_noctaves, args.grid_detect, 
                args.orb_max_features, args.star_max_size, args.star_response_threshold,
                args.star_line_threshold_binarized, args.star_suppress_nonmax_size,
                args.show]
        
        matching_options = [args.matcher, args.match_ratio, args.min_pairs, args.min_dist,
                args.max_dist, args.filter, args.min_chain_length]
        
        match_trig_options = [args.trig_group, args.method]

        optmize_options = [args.optmize_group, args.refine, args.cam_calibration]

        mre_options = [args.mre_group, args.stddev, args.initial_pose, args.strong, args.interactive]

        colocated_options = [args.colocated_group, args.min_angle]

        render_options = [args.render_group, args.texture_resolution, args.srtm, args.ground, args.direct]
        
        standard_project(args.project, args.camera, args.max_angle, detection_options, 
                matching_options, match_trig_options, optmize_options, mre_options,
                colocated_options, render_options, args.delaunay_group)