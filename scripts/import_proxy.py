# The purpose of this proxy, is to allow the import of python files that have a number at the start

create_project = __import__('1a_create_project')
globals().update(vars(create_project))

set_camera_config = __import__('1b_set_camera_config')
globals().update(vars(set_camera_config))

set_poses = __import__('2a_set_poses')
globals().update(vars(set_poses))

detect_features = __import__('3a_detect_features')
globals().update(vars(detect_features))

matching = __import__('4a_matching')
globals().update(vars(matching))

clean_and_combine_matches = __import__('4b_clean_and_combine_matches')
globals().update(vars(clean_and_combine_matches))

match_triangulation = __import__('4c_match_triangulation')
globals().update(vars(match_triangulation))

image_groups = __import__('4d_image_groups')
globals().update(vars(image_groups))

optimize = __import__('5a_optimize')
globals().update(vars(optimize))

mre_by_image = __import__('5b_mre_by_image')
globals().update(vars(mre_by_image))

colocated_feats = __import__('5b_colocated_feats')
globals().update(vars(colocated_feats))

render_model2 = __import__('6a_render_model2')
globals().update(vars(render_model2))

delaunay5 = __import__('6b_delaunay5')
globals().update(vars(delaunay5))

explore = __import__('7a_explore')
globals().update(vars(explore))