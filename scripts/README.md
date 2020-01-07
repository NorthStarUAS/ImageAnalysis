# Streamline processing of an image set with defaults:

  Run: ./process.py /path/to/images/folder

  <wait a while>

  Run: ./7a-explore.py /path/to/images/folder


# Individual scripts (the harder way)

You can also run individual scripts to execute one step at a time (or
rerun a step with different defaults.

# 1. Project Setup

Your folder of aerial images (usually one folder per flight) is
assumed to be the top level project.  This is called "the project
folder."  All the processing files will be placed inside a subfolder
called "ImageAnalysis" created inside the project folder.

You can setup, process, refine, and view your project by running the
scripts in numerical order as described below.

## 1a-create-project.py

Create the ImageAnalysis folder inside the project directory.

## 1b-set-camera-config.py

Autodetects the camera and sets up the appropriate calibration
parameters.  You can specify your camera mounting offset relative to
the airframe if needed.  If your camera is not already defined in the
database, then there is a helper script that can mostly automate this
task called: 99-new-camera.py

## 2a-set-poses.py

Specify a pix4d.csv (or sentera images-meta.txt) file to define the
aircraft location and attitude at the time each picture is taken.

Note: this project does not directly use the image gps metadata. It
additionally needs an estimate of the camera pose when the image was
taken.  There is a helper script which can automate this process:
99-make-pix4d.py

# 3. Feature Detection and Matching

  ## Feature detection algorithms:
  
  ## SIFT

  SIFT seems like the best quality detector (and feature descriptor)
  over all.  It seems to produce the most consistent overall results.
  Recommend controlling number of features by scaling the image.  For
  a DJI Phantom 4 camera, 40-50% scaling is a good starting point.
  Generally more features are better here if your computer resources
  can support it.

  If you aren't getting as many features as you like, consider a
  larger scale factor.

  ## ORB

  ORB is fast, but the feature descriptor contains less information
  and seems less robust (as compared to SIFT.)  It uses a shorter
  feature descriptor so it is more subject to noise and false matches.
  It is part of the core cv2 library so guaranteed to be available in
  all version of opencv.  Recommend 10000-20000 max features.

  ## SURF

  SURF seems to perform less well than SIFT on all fronts.  (I wrote
  this a long time ago and haven't circled back around to try this
  again more recently.)

  ## Recommendations

  Use SIFT for post processing aerial image sets when there can be
  significantly varying camera perspectives.  SIFT gives the highest
  quality at the expense of performance (and commercial license
  restrictions.)  Use ORB for real time work, gaining fast speed at
  the expense of quality (ORB performs well when processing adjacent
  video frames because the camera hasn't moved much from one frame to
  the next.)  Star may be worth trying for outdoor natural
  environments when processing one frame to the next.

  Scale: (nb) sometimes less is more!  Processing the images at scaled
  resoltuions can often be beneficial.  Reducing the scale of an image
  is a form of signal smoothing (filtering.)  All analog -> digital
  sensors have some noise, so processing your images at full 100%
  scale (--scale=1.0) may be counter productive.  Additionally feature
  detectors work (roughly) in pixel space so using a 6000x4000 image
  means most of your features will be itty bitty.  Using a 600x400
  image means most of your features will be larger items and perhaps
  more import and real.  Feature matching can get lost in the noise if
  you are matching outdoor natural environments at extreme full
  resolution.  The best scale depends on your camera, altitude, and
  even scene content.  Suggest 0.50 may be a good starting number.

  Ultimately there needs to be a balance ... too high of a
  resolution/scaling can introduce too many small indistinguishable
  features (think clumps of grass, tree leaves, corn leaves, etc.)
  But scaling the images down too much leads to not enough good
  features.  The exact amount of scaling probably depends on the
  camera, lens, altitude, and subject matter.

  ## 3a-matching.py

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

  Seems to better handle variations off plane while still being pretty
  robust and rejecting false matches.  The Essential matrix is a more
  recent discovery and encapsulates the camera calibration matrices as
  well.  Thus it can also be used to derive relative poses of the two
  images.  Results in a few percent fewer matches, but fewer outlier
  matches.  Initial optimization may also be slightly tighter.  This
  is my favorite filter right now.

  ## 3b-clean-and-combine-matches.py

  Start with the original pair-wise match set, then run several
  validation and consistency checks to remove some potential weird
  things.  The final step connects all match pairs that reference the
  same feature into match chains that can reference many images.  This
  should be run after the 4a-matching step, and can be rerun later to
  reset the matches.

  ## 3c-match-triangulation.py

  Compute an initial 3d location estimate for every feature/match
  
  ## 3d-image-groups.py

  Compute the connected groups in the image set.  Images are matched
  as pairs, but these pairs can be connected into larger groups.
  Ideally all the images will connect with each other in a giant mesh
  network, but this sadly is often not the case.  Connectivity can be
  increased by detecting more features or experimenting with different
  detectors or feature matching filters.  Ultimately, good
  connectivity is achieved by ensuring your images overlap.  70%
  overlap with neighbors is the industry recommendation.

# 4. Assemble Scene / Bundle Adjustment

  ## 4a-optimize.py

  Takes initial direct georeference scene layout and runs bundle
  adjustment.  Currently this is the best code to run.  From the
  perspective of placing the cameras correctly relative to each other,
  running the SBA optimizer with simple pair-wise features seems to
  work the best.  Features can be chained together into larger
  clusters when they match in multiple pairs, but this seems to affect
  the optimizer's ability to find a robust solution.

  ## 4c-mre-by-image.py

  Compute the mre of the assembled scene (optionally delete worst
  outliers)

  With the --interactive option, interactively display the worst mre error
  matches.  Type 'd' to delete (from matches_direct) and any other key
  to skip.  Type 'q' to quit and delete the marked matches.

  After this you will want to rerun the 4a-optimize --refine script again.

  ## 4c-colocated-feats.py

  When the vectors from a feature back to the corresponding images are
  nearly colinear, this can lead to stability and solver issues.  Very
  small changes in camera poses or calibration can move the feature
  disproportionally far distances.  The reverse is that the feature
  has to move far distances to allow small improvements in camera
  poses and this can overly constrain the solver and lead to artifacts
  in the final result.  This script sniffs these out and removes them
  if you wish.

# 6. Render Results

  ## 6a-render-model2.py

  Create a global Delaunay mesh of the optimized feature coordinates
  (like 6b-delauney5.py).  For each image in the main group, project
  the uv image grid onto the deluaney mesh to generate 3d textured
  surfaces.  This still can lead to some visible discontinuity at the
  edges of the image because it's not a dense mesh approach, but it
  generally works pretty well for many purposes.  Support filtering
  image grid squares that project too close to the horizon or fall
  outside of the convex hull of the delauney mesh.
  
  ## 6b-delaunay5.py

  Output a non-textured delaunay triangulation of the fitted surface.
  Viewable with the osgviewer utility, the model can be displayed as a
  shaded surface, wireframe, or point cloud.


# 7. Explore

  ## 7a-explore.py

  Renders the image set with all the images correctly positions.
  Allows zooming and panning.  Manages texture resolution to keep the
  current view at a high resolution while using lower resolution
  textures for surrounding areas.

  You can right click on any point to create an annotation mark.  A
  window will pop up and allow you to enter an optional comment/note.
  Right click on an existing annotation to edit or delete it.