#!/usr/bin/python

import sys

import Correlate
import Image

AffineSolver = True
SimpleSolver = False
ShutterLatencySolver = False
AttitudeBiasSolver = False
YawBiasSolver = False

defaultShutterLatency = 0.67    # measured by the shutter latency solver
defaultRollBias = -0.40         # measured by the attitude bias solver
defaultPitchBias = -1.60
defaultYawBias = -3.55          # measured by the yaw bias solver

def usage():
    print "Usage: " + sys.argv[0] + " <flight_data_dir> <raw_image_dir> <ground_alt_m>"
    exit()


# start of 'main' program
if len(sys.argv) != 4:
    usage()

flight_dir = sys.argv[1]
image_dir = sys.argv[2]
ground_alt_m = float(sys.argv[3])
geotag_dir = image_dir + "-geotag"

# create the image group
ig = Image.ImageGroup( max_features=800, detect_grid=4, match_ratio=0.5 )

# set up Samsung NX210 parameters
ig.setCameraParams(horiz_mm=23.5, vert_mm=15.7, focal_len_mm=30.0)

# load images, keypoints, descriptors, matches, etc.
ig.load( image_dir=geotag_dir )

# compute matches if needed
ig.computeMatches()
#ig.showMatches()

# correlate shutter time with trigger time (based on interval
# comaparison of trigger events from the flightdata vs. image time
# stamps.)
c = Correlate.Correlate( flight_dir, image_dir )
best_correlation, best_camera_time_error = c.test_correlations()

# tag each image with the camera position (from the flight data
# parameters) at the time the image was taken
ig.computeCamPositions(c, delay=defaultShutterLatency, 
                       rollbias=defaultRollBias, pitchbias=defaultPitchBias,
                       yawbias=defaultYawBias)

# compute a central lon/lat for the image set.  This will be the (0,0)
# point in our local X, Y, Z coordinate system
ig.computeRefLocation()

# initial placement
ig.computeKeyPointGeolocation( ground_alt_m )
print "Global error (start): %.2f" % ig.globalError()

if AffineSolver:
    for i in xrange(20):
        ig.affineTransformImages(gain=0.5)
        ig.generate_ac3d(c, ground_alt_m, geotag_dir, ref_image=None, base_name="quick-3d", version=i )
        print "Global error (%d) = %.2f: " % (i, ig.globalError())

if SimpleSolver:
    for i in xrange(20):
        ig.rotateImages(gain=0.25)
        ig.computeKeyPointGeolocation( ground_alt_m )
        ig.shiftImages(gain=0.25)
        ig.generate_ac3d(c, ground_alt_m, geotag_dir, ref_image=None, base_name="quick-3d", version=i )
        print "Global error (%d) = %.2f: " % (i, ig.globalError())

if ShutterLatencySolver:
    delaystep = 0.01
    delay = 0.5
    maxdelay = 0.7
    while delay <= maxdelay + (delaystep*0.1):
        # test fit image set with specified parameters
        ig.computeCamPositions(c, delay=delay,
                               rollbias=defaultRollBias,
                               pitchbias=defaultPitchBias,
                               yawbias=defaultYawBias)
        ig.computeKeyPointGeolocation( ground_alt_m )
        print "Global error (delay): %.2f %.2f" % (delay, ig.globalError())
        delay += delaystep

if AttitudeBiasSolver:
    pitchstep = 0.2
    minpitch = -3.0
    maxpitch = -1.0

    rollstep = 0.2
    minroll = -1.0
    maxroll = 1.0

    pitchbias = minpitch
    while pitchbias <= maxpitch + (pitchstep*0.1):
        rollbias = minroll
        while rollbias <= maxroll + (rollstep*0.1):
            # test fit image set with specified parameters
            ig.computeCamPositions(c, delay=defaultShutterLatency,
                                   rollbias=rollbias, pitchbias=pitchbias,
                                   yawbias=defaultYawBias )
            ig.computeKeyPointGeolocation( ground_alt_m )
            print "Global error (attitude): %.2f %.2f %.2f" % (pitchbias, rollbias, ig.globalError())
            rollbias += rollstep
        pitchbias += pitchstep

if YawBiasSolver:
    yawstep = 0.1
    minyaw = -5.0
    maxyaw = -3.0

    yawbias = minyaw
    while yawbias <= maxyaw+ (yawstep*0.1):
        # test fit image set with specified parameters
        ig.computeCamPositions(c, delay=defaultShutterLatency,
                               rollbias=defaultRollBias,
                               pitchbias=defaultPitchBias,
                               yawbias=yawbias )
        ig.computeKeyPointGeolocation( ground_alt_m )
        print "Global error (yaw): %.2f %.2f" % (yawbias, ig.globalError())
        yawbias += yawstep
