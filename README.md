# Image Analysis

Aerial imagery analysis, processing, and presentation scripts.

## 3rd_party

   Home to 3rd party code that needs modifications or adjustments to
   be helpful.  (Or things that aren't commonly available
   system-wide.)

## ils

   This relates to systems that have an illumniation sensor pointing
   up at the sky.  When the drone pitches or rolls for turning or
   forward motion, the sensor no longer points up.  This code
   understands date, time, location, as well as the relative sun
   location and angle.  It attempts to correct the illumination sensor
   for attitude errors and thus produce more consistant results in the
   output images.

## lib

   library of code, mostly to support feature detection, feature
   matching, and scene assembly.

## movie

   Some of these image analysis techniques can be applied to movies in
   interesting ways.

   ### Use feature matching between consecutive movie frames to
       accurately estimate a gyro axis (aligned with the camera center
       of projection.)  Will also estimate the 2nd and 3rd gyro axes,
       but with less quality.

   ### Track Aruco codes.  If you have control over your scene and can
       place Aruco codes in strategic places, they are extremely
       awesome.  The detection code is extremely fast, very reliable,
       each marker has it's own code, and all 4 corners of the marker
       are identified.

   ### Extract still shots (frames) from a movie and geotag them.

   ### Generate an augmented reality hud overlay on top of an
       in-flight movie.

## scripts

   A series of front-end scripts that primarily pair with the lib
   directory for feature detection, matching, and scene assembly.

## srtm

   For the purposes of generating an initial earth surface estimate,
   use SRTM data.  Given a camera pose estimate (i.e. from flight
   data) and an earth surface estimate, we can project out the feature
   vectors and estimate their 3d locations.  This is useful for
   generating an initial guess to feed the sparse optimizer (sparse
   bundle adjustment.)

## tests

   A random collection of scripts for testing different things.