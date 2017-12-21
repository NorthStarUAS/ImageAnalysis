# Image Analysis

Aerial imagery analysis, processing, and presentation scripts.

## Urgent plea!

   If anyone sees this and knows of a forum or mailing list or some
   other discussion group that focuses on image stitching and bundle
   adjustment, please contact me and let me know where to find it!

   It feels like the experts in the field fall into one of the
   following groups:

   - As soon as they gain some knowledge, they are snatched up by a
     big company and are unable to talk about any of this technology.

   - They became an expert for their phd work or some research
     project, but now it's 10 years later and they have different
     things to be interested in.

   - As soon as they get something to work, they spin off a commercial
     venture and are busy with their new business.

   So if you are reading this and interested in this field, talk to
   me!  Where do people discuss these topics?  Should we start our own
   forum?
   
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