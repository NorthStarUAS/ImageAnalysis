# 1. Project Setup

## 1a-create-project.py

Create a project directory and stub config files

## 1b-set-camera-config.py

Define camera calibration and lens distortion parameters.

## 1c-import-images.py

Import images into project work space (may optionally scale the images for
faster processing and less memory/disk space usage.)

## 1d-load-images.py

Tests loading of image info, then ends.

## 1e-show-images.py

Show all the imported images.


# 2. Set Initial Camera Poses

# 3. Feature Detection

  ## SIFT

  SIFT seems like the best quality detector (and feature descriptor)
  over all.  It seems to produce the most consistent overall results.
  Recommend 5000-10000 sift features per image.  More is better if
  your computer resources can support it.

  ## ORB

  ORB is fast, but the feature descriptor contains less information
  and seems less robust (as compared to SIFT.)  It doesn't seem to
  find as many valid matches between image pairs.  It is part of the
  core cv2 library so guaranteed to be available in all version of
  opencv.  Recommend 10000-20000 max features.

  ## Star (CenSuRE)

  Star is proposed to more consistently find features in natural
  environments.  However, these features may not be 'better' as
  defined in an end-to-end perspective for the whole pipeline.  At
  least Star produces a more even distribution of features across an
  image.  Typically trees next to a field suck up all the attention of
  the feature detector where as the field is ultimate what we wish the
  detector would focus on.

  Possibly doesn't work as well with matching images taken at
  different orientations?  I see group connectivity issues that imply
  CenSuRE is orientation sensitive.  It might be good for outdoor SLAM
  type applications.

  No longer in the core cv2 library as of opencv 3.x
   
  ## SURF

  SURF seems to perform less well than SIFT on all fronts.

  ## Recommendations

  Use SIFT for post processing aerial image sets when there can be
  significantly varying camera perspectives.  SIFT gives the highest
  quality at the expense of performance (and commercial license
  restrictions.)  Use ORB for real time work, gaining fast speed at
  the expense of quality (ORB does perform really well when processing
  video when the camera hasn't moved much from one frame to the next.)
  Star may be worth trying for outdoor natural environments when
  processing one frame to the next.

# 4. Feature Matching

  ## 4a-matching.py

  Run this script to find all the matching feature pairs in the image set.
  
    ### Homography filter

    Enforces a planer relationship between features matched in a pair
    of images.  Features off that plane tend to be rejected.  This
    seems to be the recommended filter relationship in all the
    examples and literature.

    ### Fundamental filter

    The homography matrix can only map points on a plane in one image
    to points on a plane in another images.  The fundamental filter is
    more flexible with depth (but could pass more noise through, but
    this noise might be more filterable as mean reprojection error?)
    We know from epipolar geomtry that any feature in one image will
    map onto some line in the second image.  The fundamental matrix
    captures this relationship.  Bad matches can still slip through if
    they lie on this line.
    
    ### Essential filter

    Seems to better handle variations off plane while still being
    pretty robust and rejecting false matches.  (Only available with
    OpenCV3?)  The Essential matrix is a more recent discovery and
    encapsulates the camera calibration matrices as well.  Thus it can
    also be used to derive relative poses of the two images.

  ## 4b-clean-and-reset-matches.py

  Start with the original pair-wise match set, then run several
  validation and consistency checks to remove some potential weird
  things.  This should be run after the 4a-matching step, and can be
  rerun later to reset the matches.
  
  ## 4b-simple-matches-reset.py

  Start with the original pair-wise match set and regenerate a
  matches_direct version of this with no extra checking or validation.

  ## 4c-match-grouping.py

  Iterate through the matches_direct structure, locate and connect all
  match chains.  Continues until no more chains can be connected.
  
  ## 4d-image-groups.py

  Images are matches as pairs, but these pairs can be connected into
  larger groups.  Ideally all images will connect with each other, but
  this sadly is often not the case.  Connectivity can be increased by
  detecting more features or experimenting with different detectors or
  feature matching filters.  Ultimately, good connectivity is achieved
  by ensuring your images overlap.  70% overlap with neighbors is the
  industry recommendation.

  ## 4e-review-matches.py

  Interactive script that presents the worst outliers (based on some
  relationship fit of the match pairs) and allows the user to review
  and accept or delete the match.  

# 5. Assemble Scene / Bundle Adjustment

  ## 5a-sba1.py

  Original attempt, deprecated in favor of 5a-sba2.py

  ## 5a-sba2.py (best so far)

  Takes initial direct georeference scene layout and runs bundle
  adjustment.  Currently this is the best code to run.  From the
  perspective of placing the cameras correctly relative to each other,
  running the SBA optimizer with simple pair-wise features seems to
  work the best.  Features can be chained together into larger
  clusters when they match in multiple pairs, but this seems to affect
  the optimizer's ability to find a robust solution.

  ## 5a-sba3.py

  Uses image connectivity structure combined with the ransac version
  of solvepnp() and a custom multi-vector triangulation function to
  assemble scene.  Seems to produce a useful initial scene structure
  for the set of images that are well connected together.  The ransac
  version of solvepnp() helps the robustness of the assembly process.

  ## 5a-sba4.py

  Test of opencv3's findEssentialMat() and recoverPose() to build the
  initial scenery construction.  Given all the pairwise relative
  rotations, send to optimizer to find an optimal set of camera
  orientations that minimize the slop.

  ## 5c-mre-by-feature2.py

  Compute the mre of the assembled scene (optionally delete worst
  outliers)

  With the --interactive option, interactively display the worst mre error
  matches.  Type 'd' to delete (from matches_direct) and any other key
  to skip.  Type 'q' to quit and delete the marked matches.

  After this you will want to rerun the 4a-groups.py, then 5a-sba2.py
  script to reoptimize the fit.


# 6. Render Results

  ## 6b-delaunay3.py

  Current best script for textured output

  ## 6b-delaunay5.py

  Output a non-textured delaunay triangulation of the fitted surface
