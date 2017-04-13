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

  ORB is fast, but doesn't seem to be as widely robust as SIFT.  It
  doesn't seem to find as many valid matches between image pairs.  It
  is part of the core cv2 library so guaranteed to be available in all
  version of opencv.  Recommend 10000-20000 max features.

  ## Star (CenSuRE)

  Star is proposed to more consistently find features in natural
  environments.  However, these features may not be 'better' as
  defined in an end-to-end perspective for the whole pipeline.  At
  least Star produces a more even distribution of features across an
  image.  Typically trees next to a field suck up all the attention of
  the feature detector where as the field is ultimate what we wish the
  detector would focus on.

  Possible downside with matching images of different orientations?
  Group connectivity issues at least.

  No longer in the core cv2 library as of opencv 3.x
   
  ## SURF

  SURF seems to perform less well than SIFT on all fronts.

  ## Recommendations

  Use SIFT for post processing aerial image sets when there can be
  significantly varying camera perspectives.  SIFT gives the highest
  quality at the expense of performance.  Use ORB for real time work,
  gaining fast speed at the expense of quality (ORB does perform
  really well when processing video when the camera hasn't moved much
  from one frame to the next.)

# 4. Feature Matching

  ## Homography filter

  Enforces a planer relationship between features matched in a pair
  of images.  Features off that plane tend to be rejected.

  ## Essential filter

  Seems to better handle variations off plane while still being pretty
  robust and rejecting false matches.

  ## Group connectivity.

  Images are matches as pairs, but these pairs can be connected into
  larger groups.  Ideally all images will connect with each other, but
  this sadly is often not the case.  Connectivity can be increased by
  detecting more features or experimenting with different detectors or
  feature matching filters.
  
# 5. Assemble Scene / Bundle Adjustment

  ## 5a-sba1.py

  Original attempt, deprecated in favor of 5a-sba2.py

  ## 5a-sba2.py

  Takes initial direct georeference scene layout and runs bundle
  adjustment.  Currently this is the best code to run.  From the
  perspective of placing the cameras correctly relative to each other,
  running the SBA optimizer with simple pair-wise features seems to
  work the best.  Features can be chained together into larger
  clusters when they match in multiple pairs, but this seems to affect
  the optimizer's ability to find a robust solution.

  ## 5a-sba3.py

  Uses image connectivity structure combined with solvepnp() and a
  custom multi-vector triangulation function to assemble scene.  Works
  well until we run out of sufficient connectivity.

  ## 5a-sba4.py

  Test of opencv3's findEssentialMat() and recoverPose() to build the
  initial scenery construction.  Given all the pairwise relative
  rotations, send to optimizer to find an optimal set of camera
  orientations that minimize the slop.

## 5c-mre-by-feature3.py

Compute the mre of the assembled scene (optionally delete worst outliers)

With the --show option, interactively display the worst mre error
matches.  Type 'y' to delete (from matches_direct) and any other key
to skip.  Type 'q' to quit and delete the marked matches.


# 6. Render Results

  ## 6a-delaunay3.py

  Current best script for textured output

  ## 6a-delaunay5.py

  Output a non-textured delaunay triangulation of the fitted surface