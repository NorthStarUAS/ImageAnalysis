#!/usr/bin/python

import sys

# find our custom built opencv first
sys.path.insert(0, "/home/curt/Projects/ComputerVision/lib/python2.7/site-packages/")

import FlightData
import ImageGroup
import Matcher
import Solver

ComputeMatches = False
EstimateGroupBias = False
EstimateCameraDistortion = False
ReviewMatches = False

SpecialReview = []
# SpecialReview = [ "SAM_0021.JPG", "SAM_0022.JPG", "SAM_0023.JPG", "SAM_0024.JPG", "SAM_0026.JPG", "SAM_0037.JPG", "SAM_0056.JPG", "SAM_0075.JPG", "SAM_0077.JPG", "SAM_0093.JPG", "SAM_0104.JPG", "SAM_0113.JPG", "SAM_0114.JPG", "SAM_0115.JPG", "SAM_0122.JPG" ]

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
m = Matcher.Match(image_group=ig)
if ComputeMatches:
    m.computeMatches()
    m.addInverseMatches()
    #m.showMatches()

# now compute the keypoint usage map
ig.genKeypointUsageMap()

# reset fit to start from scratch...
#ig.zeroImageBiases()

# correlate shutter time with trigger time (based on interval
# comaparison of trigger events from the flightdata vs. image time
# stamps.)
c = FlightData.Correlate( flight_dir, image_dir )
best_correlation, best_camera_time_error = c.test_correlations()

# tag each image with the camera position (from the flight data
# parameters) at the time the image was taken
ig.computeCamPositions(c, force=False, weight=True)
#ig.generate_camera_location_report()

# weight the images (either automatically by roll/pitch, or force a value)
ig.computeWeights(force=1.0)

# compute a central lon/lat for the image set.  This will be the (0,0)
# point in our local X, Y, Z coordinate system
ig.computeRefLocation()

# initial projection
ig.k1 = -0.00028
ig.k2 = 0.0
ig.projectKeypoints(do_grid=True)

# review matches
if len(SpecialReview):
    e = ig.groupError()
    print "Group error (start): %.2f" % e
    for name in SpecialReview:
        ig.reviewImageErrors(name, minError=0.001)
        ig.saveMatches()

if ReviewMatches:
    e = ig.groupError()
    print "Group error (start): %.2f" % e
    m.reviewImageErrors(minError=1.0)
    m.saveMatches()
    # re-project keypoints after outlier review
    ig.projectKeypoints()

e = ig.groupError(method="average")
stddev = ig.groupError(method="variance")
print "Group error (start): %.2f" % e
print "Group standard deviation (start): %.2f" % stddev

s = Solver.Solver(image_group=ig, correlator=c)
#s.AffineFitter(steps=1, gain=0.4, fullAffine=False)

if EstimateGroupBias:
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

if EstimateCameraDistortion:
    s.estimateParameter("k1", -0.005, 0.005, 0.001, 3)
    s.estimateParameter("k2", -0.005, 0.005, 0.001, 3)

for i in xrange(0):
    # minimize error variance (tends to align image orientation)
    ig.fitImagesIndividually(method="variance", gain=0.2)
    ig.projectKeypoints(do_grid=True)
    e = ig.groupError(method="average")
    stddev = ig.groupError(method="variance")
    print "Variance fit, iteration = %d" % i
    print "  Group error: %.2f" % e
    print "  Group standard deviation: %.2f" % stddev

    # find x & y shift values to minimize placement

    # we already have gain on our variance adjustment so if we
    # multiply gain again here we will walk our x, y shifts very
    # slowly
    ig.shiftImages(gain=1.0)
    ig.projectKeypoints(do_grid=True)
    e = ig.groupError(method="average")
    stddev = ig.groupError(method="variance")
    print "Shift fit, iteration = %d" % i
    print "  Group error: %.2f" % e
    print "  Group standard deviation: %.2f" % stddev

    # minimize overall error (without moving camera laterally)
    ig.fitImagesIndividually(method="direct", gain=0.4)
    ig.projectKeypoints(do_grid=True)
    e = ig.groupError(method="average")
    stddev = ig.groupError(method="variance")
    print "Direct fit, iteration = %d" % i
    print "  Group error: %.2f" % e
    print "  Group standard deviation: %.2f" % stddev


if False:
    name = "SAM_0032.JPG"
    image = ig.findImageByName(name)
    #ig.render_image(image, cm_per_pixel=15.0, keypoints=True)

    image_list = []
    image_list.append(name)
    for i, pairs in enumerate(image.match_list):
        if len(pairs) < 3:
            continue
        image_list.append(ig.image_list[i].name)

if True:
    #image_list = [ "SAM_0031.JPG", "SAM_0032.JPG", "SAM_0033.JPG" ] 
    #image_list = [ "SAM_0153.JPG", "SAM_0154.JPG", "SAM_0155.JPG", "SAM_0156.JPG", "SAM_0157.JPG", "SAM_0158.JPG" ] 
    image_list = [ "SAM_0184.JPG", "SAM_0185.JPG", "SAM_0186.JPG", "SAM_0187.JPG", "SAM_0188.JPG", "SAM_0189.JPG", "SAM_0190.JPG" ] 
    print str(image_list)
    for name in image_list:
        image = ig.findImageByName(name)
        ig.findImageShift(image, gain=1.0, placing=True)
        image.placed = True
    ig.render_image_list(image_list, cm_per_pixel=10.0, keypoints=True)

#ig.placeImages()
s.AffineFitter(steps=30, gain=0.4, fullAffine=False)
