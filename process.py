#!/usr/bin/python

import sys

import FlightData
import ImageGroup
import Solver

EstimateGlobalBias = True
ReviewMatches = False

# values for flight: 2014-06-06-01
#defaultShutterLatency = 0.66    # measured by the shutter latency solver
#defaultRollBias = -0.88          # measured by the roll bias solver
#defaultPitchBias = -1.64         # measured by the pitch bias solver
#defaultYawBias = -5.5           # measured by the yaw bias solver
#defaultAltBias = -8.9           # measured by the alt bias solver ...

# values for flight: 2014-06-06-02
#defaultShutterLatency = 0.63    # measured by the shutter latency solver
#defaultRollBias = -0.84         # measured by the roll bias solver
#defaultPitchBias = 0.40         # measured by the pitch bias solver
#defaultYawBias = 2.84           # measured by the yaw bias solver
#defaultAltBias = -9.52         # measured by the alt bias solver ...

# values for flight: 2014-05-28
#defaultShutterLatency = 0.66    # measured by the shutter latency solver
#defaultRollBias = 0.0          # measured by the roll bias solver
#defaultPitchBias = 0.0         # measured by the pitch bias solver
#defaultYawBias = 0.0           # measured by the yaw bias solver
#defaultAltBias = 0.0           # measured by the alt bias solver ...

def usage():
    print "Usage: " + sys.argv[0] + " <flight_data_dir> <raw_image_dir> <ground_alt_m>"
    exit()


# start of 'main' program
if len(sys.argv) != 4:
    usage()

flight_dir = sys.argv[1]
image_dir = sys.argv[2]
ground_alt_m = float(sys.argv[3])
work_dir = image_dir + "-work"

# create the image group
ig = ImageGroup.ImageGroup( max_features=800, detect_grid=4, match_ratio=0.5 )

# set up Samsung NX210 parameters
ig.setCameraParams(horiz_mm=23.5, vert_mm=15.7, focal_len_mm=30.0)

# set up World parameters
ig.setWorldParams(ground_alt_m=ground_alt_m)

# load images, keypoints, descriptors, matches, etc.
ig.update_work_dir(source_dir=image_dir, work_dir=work_dir)
ig.load()

# compute matches if needed
ig.computeMatches()
#ig.showMatches()

# correlate shutter time with trigger time (based on interval
# comaparison of trigger events from the flightdata vs. image time
# stamps.)
c = FlightData.Correlate( flight_dir, image_dir )
best_correlation, best_camera_time_error = c.test_correlations()

# tag each image with the camera position (from the flight data
# parameters) at the time the image was taken
ig.computeCamPositions(c)

# compute a central lon/lat for the image set.  This will be the (0,0)
# point in our local X, Y, Z coordinate system
ig.computeRefLocation()

# initial projection
ig.projectKeypoints()

# review matches
if ReviewMatches:
    e = ig.globalError()
    print "Global error (start): %.2f" % e
    ig.reviewImageErrors(minError=10.0)
    ig.saveMatches()

# re-project keypoints after outlier removal
ig.projectKeypoints()
e = ig.globalError()
print "Global error (start): %.2f" % e

s = Solver.Solver(image_group=ig, correlator=c)

if EstimateGlobalBias:
    # parameter estimation can be slow, so save our work after every
    # step
    s.estimateParameter("shutter-latency", 0.5, 0.7, 0.1, 3)
    ig.save_project()
    s.estimateParameter("yaw", -10.0, 10.0, 2.0, 3)
    ig.save_project()
    s.estimateParameter("roll", -5.0, 5.0, 1.0, 3)
    ig.save_project()
    s.estimateParameter("pitch", -5.0, 5.0, 1.0, 3)
    ig.save_project()
    s.estimateParameter("altitude", -20.0, 0.0, 2.0, 3)
    ig.save_project()

for i in xrange(5):
    ig.fitImagesIndividually(gain=0.5)
    ig.projectKeypoints()
    print "Global error (after individual fit): %.2f" % ig.globalError()

s.AffineFitter(steps=20, gain=0.5, fullAffine=False)
