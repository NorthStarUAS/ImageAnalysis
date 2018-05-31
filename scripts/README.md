# 1. Project Setup

## 1a-create-project.py

Create a project directory, setup config files, define source path of
image set.

## 1b-set-camera-config.py

Define camera calibration, lens distortion parameters, aircraft
mounting offset.

## 1d-load-images.py

Tests loading of image info, then ends.

## 1e-show-images.py

Show all the images.

# 2. Set Initial Camera Poses

## 2a-set-aircraft-poses.py

Specify a pix4d.csv (or sentera images-meta.txt) file to define the
aircraft location and attitude at the time each picture is taken.

## 2b-set-camera-poses.py

Apply the camera mounting offset to estimate the actual camera pose
for each image.

# 3. Feature Detection

  ## SIFT

  SIFT seems like the best quality detector (and feature descriptor)
  over all.  It seems to produce the most consistent overall results.
  Recommend 10000-20000 sift features per image.  Generally more is
  better here if your computer resources can support it.

  If you aren't getting as many features as you ask for, consider a
  larger scale factor.

  ## ORB

  ORB is fast, but the feature descriptor contains less information
  and seems less robust (as compared to SIFT.)  It uses a shorter
  feature descriptor so it is more subject to noise and false matches.
  It is part of the core cv2 library so guaranteed to be available in
  all version of opencv.  Recommend 10000-20000 max features.

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

  Important note: sometimes less is more!  Processing the images at
  scaled resoltuions can often be beneficial.  Feature detectors work
  (roughly) in pixel space so using a 6000x4000 image means most of
  your features will be itty bitty.  Using a 600x400 image means most
  of your features will be larger items and perhaps more import and
  real.  Feature matching can get lost in the noise if you are
  matching outdoor natural environments at extreme full resolution.

  Ultimately there needs to be a balance ... too high of a
  resolution/scaling can introduce too many small indistinguishable
  features (think clumps of grass, tree leaves, corn leaves, etc.)
  But scaling the images down too much leads to not enough good
  features.  The exact amount of scaling probably depends on the
  camera, lens, altitude, and subject matter.

# 4. Feature Matching

  ## 4a-matching.py

  Run this script to find all the matching feature pairs in the image set.

    ### GMS filter ###

    This seems really useful: http://jwbian.net/gms

    It uses gridded motion statistics to find the valid match set and
    seems to much more robust to noise and perform much better than
    the traditional homography filter.  Often homography rejects many
    good matches because of the amount of noise/outliers in the
    potential match set.

    Currently this is my favorite approach, it oftens seems to find
    too many valid matches if that is possible.

    Paired with the traditional sift distance ratio test <plus> a
    first pass to discard matches with > average match distance <plus>
    discarding any matches that don't exist in the reciprocal set ==
    very few false matches, but still enough good matches.
    
    ### Homography filter

    Enforces a planer relationship between features matched in a pair
    of images.  Features off that plane tend to be rejected.  This
    seems to be the recommended filter relationship in all the
    examples and literature.  This is the filter that tends to show up
    in all the feature matching tutorials, however it doesn't perform
    well if your features aren't planar (think trees, terrain,
    structures, etc.)

    ### Fundamental filter

    The homography matrix can only map points on a plane in one image
    to points on a plane in another images.  The fundamental filter is
    more flexible with depth (but could pass more noise through, but
    this noise might be more filterable as mean reprojection error?)
    We know from epipolar geomtry that any feature in one image will
    map onto some line in the second image.  The fundamental matrix
    captures this relationship.  Bad matches can still slip through if
    they lie on this line.
    
    ### Essential filter (*)

    Seems to better handle variations off plane while still being
    pretty robust and rejecting false matches.  (Only available with
    OpenCV3?)  The Essential matrix is a more recent discovery and
    encapsulates the camera calibration matrices as well.  Thus it can
    also be used to derive relative poses of the two images.  Results
    in a few percent fewer matches, but fewer outlier matches.
    Initial optimization may also be slightly tighter.  This is my
    favorite filter right now.

  ## 4b-clean-and-reset-matches.py

  Start with the original pair-wise match set, then run several
  validation and consistency checks to remove some potential weird
  things.  This should be run after the 4a-matching step, and can be
  rerun later to reset the matches.
  
  ## 4b-simple-matches-reset.py

  Start with the original pair-wise match set and regenerate a
  matches_direct version of this with no extra checking or validation.
  You shouldn't ever need/want to run this script under normal
  circumstances.

  ## 4c-review-matches.py

  This should be done on the original match pairs before full chain
  grouping to hopefully avoid incorrectly linking chains due to a bad
  match pair.

  This is an interactive script that compares pair-wise matches and
  presents the worst outliers (based on the homography or affine
  relationship of the matches pairs) and allows the user to review and
  accept or delete potential outliers.

  Make sure to re-run 4d-match-grouping.py after 4c-review-matches.py
  if you return and review more matches later.

  ## 4d-match-grouping.py

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

# 5. Assemble Scene / Bundle Adjustment

  ## 5a-optimize.py

  Takes initial direct georeference scene layout and runs bundle
  adjustment.  Currently this is the best code to run.  From the
  perspective of placing the cameras correctly relative to each other,
  running the SBA optimizer with simple pair-wise features seems to
  work the best.  Features can be chained together into larger
  clusters when they match in multiple pairs, but this seems to affect
  the optimizer's ability to find a robust solution.

  Note 1: it seems productive to optimize the fully grouped (linked
  group chains) version of the match file versus the pair-wise match
  file.

  Note 2: it seems productive to place bounds on feature and camera
  locations.  This seems to improve convergence speed in the first few
  steps, but slows convergence through the trailing steps.  It also
  may increase the error in the final fit; however, the result is
  probably more correct considering a better fit requires moving
  points to impossible places.)


  ## 5c-mre-by-feature3.py

  Compute the mre of the assembled scene (optionally delete worst
  outliers)

  With the --interactive option, interactively display the worst mre error
  matches.  Type 'd' to delete (from matches_direct) and any other key
  to skip.  Type 'q' to quit and delete the marked matches.

  After this you will want to rerun the 4a-groups.py, then 5a-sba2.py
  script to reoptimize the fit.


# 6. Render Results

  ## 6a-render-model1.py

  Bins the covered region into a grid.  Iterates through the fitted
  features and sorts them into corresponding x,y bins.  Find the
  average elevation per bin.  Then fill in the empty bins based on
  average of neighbor bin elevations.

  For each image, create an 8 by 8 grid model and project this grid
  onto the binned surface.
  
  Optionally can project onto a flat plane or the SRTM surface, but
  surface inaccuracies lead to edge mismatches.  This is where dense
  mesh construction has an advantage, but the dense mesh may not be
  the best tool for every application.

  The quality of this result hinges on the accuracy of the surface
  approximation versus the true surface.

  ## 6a-render-model2.py

  Loosely follow the approach of 6b-delauney5.py.  Create a per image
  Delaunay mesh of fitted features.  This mesh will be a convex hull
  of features found/matched within each imaage.  Also include proper
  textures and texture coordinates.  This should give accurate texture
  placement and surface elevations for the points in the sparse mesh.

  Outside of the convex hull, generate a secondary model to cover the
  remaining area of the image ... project the corners/edges onto a
  surface approximation and alpha blend away the extreme edges.
  
  
  ## 6b-delaunay3.py

  Insert description

  ## 6b-delaunay5.py

  Output a non-textured delaunay triangulation of the fitted surface


# 7. Explore

  ## 7a-explore.py

  Renders the image set with all the images correctly positions.
  Allows zooming and panning.  Manages texture resolution to keep the
  current view at a high resolution while using lower resolution
  textures for surrounding areas.

  The plan: add support for marking / labeling / saving features or
  areas of interest.  I would like to be able to import/export shape
  files.  I would like to be able to run various pre-filtering
  algorithms on the images to help highlight points of interest (use
  case dependent.)