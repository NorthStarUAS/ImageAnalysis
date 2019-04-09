# Image Analysis

Aerial imagery analysis, processing, and presentation scripts.

# Brief Overview

This image analysis toolset started as two independent efforts to
build a better (faster, free-er, more robust) image stitching system
for aerial surveying compared to what was commercially available.  As
a consequence of work, location, and project focus changes, portions
of these two efforts were combined and then a large amount of
subsequent work has been performed.  'A lot' has been learned in the
mean time, not limited to: how hard the problem is, how good the
commercial tools actual are, mastering many techniques and strategies
related to the underlying task of image stitching, applying these
techniques to some loosely related, but interesting problems.

The project goals remain:

- develop a quality open-source image fitting (stitching) and scene
  reconstruction tool chain that is appropriate for examination
  (education) and modification (research use.)

- leverage well known building blocks including python, opencv, numpy,
  and scipy.

- continue to develop and refine offshoot projects that support our
  UAS Lab projects: such as EKF validation, augmented reality, movie +
  flight data time correlation.

- develop extensions and improvements in support of ongoing research
  project.  (Such as: invasive species surveys, mapping, and ground
  truthing.)

# Road Map

Briefly, near term development goals include:

- [done] Renable/retool code to best fit optimized solution to original
  camera pose locations.  (This ensures the solution didn't walk off
  in some random direction, and also that the final fit is as close to
  the real world location as we are able to do with the given data
  set.)

- [done] Continued improvement of python3 support

- [first release available] A scene exploration tool that knows all
  the views covering any point, can geolocate any selected point in an
  image, can import/export shape files or feature databases for map
  creation or validation.

Medium term development goals include:

- Continued exploration of machine vision strategies for automatically
  identifying/locating items of interest in imagery.  (Such as:
  oriental bittersweet.)

Wish list items:

- The current code optimizes the 'sparse' mesh, but does not do dense
  mesh reconstruction.  It would be interesting to explore dense mesh
  generation for creating detailed DEM's and exact orthophotos.  This
  is a core strength of commercial image stitching tools, but perhaps
  not necessary for every aerial survey application.

  The most important objective is to answer questions like "Where is
  all the oriental bittersweet in the survey area?"  A pretty picture,
  an orthorphoto, or a DEM may not always be the best tool to help
  answer that question, while other customized approaches may do
  better.

- I would love to see a community of interested tinkerers grow up
  around this project.  I think there are many important concepts and
  strategies here that have far ranging uses and implications.  It
  would be great to keep a light shining on this area so the expertise
  doesn't get entirely buried inside the caves of the commercial
  world.


# Code Layout

## 3rd_party

   Home to 3rd party code that needs modifications or adjustments to
   be helpful.  (Or things that aren't commonly available
   system-wide.)

## scripts

   A series of front-end scripts that primarily pair with the lib
   directory for feature detection, matching, and scene assembly.
   
## scripts/lib

   Libraries of python modules with all the functionality that
   supports the higher level scripts.


## srtm

   For the purposes of generating an initial earth surface estimate,
   use SRTM data.  Given a camera pose estimate (i.e. from flight
   data) and an earth surface estimate, we can project out the feature
   vectors and estimate their 3d locations.  This is useful for
   generating an initial guess to feed the sparse optimizer (sparse
   bundle adjustment.)

## tests

   A random collection of scripts for testing different things.

   - ils: This relates to systems that have an illumination sensor
     pointing up at the sky.  When the drone pitches or rolls for
     turning or forward motion, the sensor no longer points up.  This
     code understands date, time, location, as well as the relative
     sun location and angle.  It attempts to correct the illumination
     sensor for attitude errors and thus produce more consistent
     results in the output images.

## video

   Some of these image analysis techniques can be applied to movies in
   interesting ways.

   - Use feature matching between consecutive movie frames to
     accurately estimate a gyro axis (aligned with the camera center
     of projection.)  Will also estimate the 2nd and 3rd gyro axes,
     but with less quality.

   - Track Aruco codes.  If you have control over your scene and can
     place Aruco codes in strategic places, they are extremely
     awesome.  The detection code is extremely fast, very reliable,
     each marker has it's own code, and all 4 corners of the marker
     are identified.

   - Extract still shots (frames) from a movie and geotag them.

   - Generate an augmented reality hud overlay on top of an in-flight
     movie.

## installation hints

  - sudo pip3 install opencv-contrib-python
  - sudo pip3 install --pre --extra-index-url https://archive.panda3d.org/ panda3d