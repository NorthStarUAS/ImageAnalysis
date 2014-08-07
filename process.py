#!/usr/bin/python

import sys

# find our custom built opencv first
sys.path.insert(0, "/home/curt/Projects/ComputerVision/lib/python2.7/site-packages/")

import FlightData
import ImageGroup
import Solver

ComputeMatches = True
EstimateGroupBias = False
EstimateCameraDistortion = False
ReviewMatches = False
ReviewPoint = (-81.34096, 27.65065)
FitIterations = 1

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
ig = ImageGroup.ImageGroup( max_features=800, detect_grid=4, match_ratio=0.75 )

# set up Samsung NX210 parameters
ig.setCameraParams(horiz_mm=23.5, vert_mm=15.7, focal_len_mm=30.0)

# set up World parameters
ig.setWorldParams(ground_alt_m=ground_alt_m)

# load images, keypoints, descriptors, matches, etc.
ig.update_work_dir(source_dir=image_dir, work_dir=work_dir, scale=0.20)
ig.load()

# compute matches if needed
if ComputeMatches:
    ig.m.robustGroupMatches(method="fundamental", review=False)
    ig.m.cullMatches()
    ig.m.addInverseMatches()
    #ig.m.showMatches()

# now compute the keypoint usage map
ig.genKeypointUsageMap()

# reset fit to start from scratch...
#ig.zeroImageBiases()

# correlate shutter time with trigger time (based on interval
# comaparison of trigger events from the flightdata vs. image time
# stamps.)
c = FlightData.Correlate( flight_dir, image_dir )
best_correlation, best_camera_time_error = c.test_correlations()

# compute the aircraft position (from the flight data parameters) at
# the time the image was taken
ig.interpolateAircraftPositions(c, force=False, weight=True)
#ig.generate_aircraft_location_report()

# compute a central lon/lat for the image set.  This will be the (0,0)
# point in our local X, Y, Z coordinate system
ig.computeRefLocation()

# compute initial camera poses based on aircraft pose + pose biases +
# camera mounting offset + mounting biases
ig.estimateCameraPoses(force=False)

# weight the images (either automatically by roll/pitch, or force a value)
ig.computeWeights(force=1.0)
ig.computeConnections()

# initial projection
ig.k1 = -0.00028
ig.k2 = 0.0
#ig.recomputeWidthHeight()
ig.projectKeypoints(do_grid=True)

#ig.sfm_test()
#ig.pnp_test()

# review matches
if ReviewMatches:
    e = ig.groupError(method="average")
    stddev = ig.groupError(method="stddev")
    print "Group error (start): %.2f" % e
    ig.m.reviewImageErrors(minError=(e+stddev))
    ig.m.saveMatches()
    # re-project keypoints after outlier review
    ig.projectKeypoints()

#if ReviewPoint != None:
#    ig.m.reviewPoint( ReviewPoint[0], ReviewPoint[1], ig.ref_lon, ig.ref_lat)

if False:
    # recompute matches with a bit larger match ratio than default
    ig.m.match_ratio *= 1.4
    i1 = ig.m.findImageByName("SAM_0369.JPG")
    i1.match_list = ig.m.computeImageMatches(i1, review=True)
    i1.save_matches()

e = ig.groupError(method="average")
stddev = ig.groupError(method="stddev")
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

OldFit = False
if OldFit:
    # fit by sweeping the camera pose through ranges of motion and finding
    # the values that minimize the error
    for i in xrange(FitIterations):
        # minimize error stddev (tends to align image orientation)
        ig.fitImagesIndividually(method="stddev", gain=0.2)
        ig.projectKeypoints(do_grid=True)
        e = ig.groupError(method="average")
        stddev = ig.groupError(method="stddev")
        print "Standard deviation fit, iteration = %d" % i
        print "  Group error: %.2f" % e
        print "  Group standard deviation: %.2f" % stddev

        # find x & y shift values to minimize placement

        # we already have gain on our stddev adjustment so if we
        # multiply gain again here we will walk our x, y shifts very
        # slowly
        ig.shiftImages(gain=1.0)
        ig.projectKeypoints(do_grid=True)
        e = ig.groupError(method="average")
        stddev = ig.groupError(method="stddev")
        print "Shift fit, iteration = %d" % i
        print "  Group error: %.2f" % e
        print "  Group standard deviation: %.2f" % stddev

        # minimize overall error (without moving camera laterally)
        ig.fitImagesIndividually(method="average", gain=1.0)
        ig.projectKeypoints(do_grid=True)
        e = ig.groupError(method="average")
        stddev = ig.groupError(method="stddev")
        print "Average error fit, iteration = %d" % i
        print "  Group error: %.2f" % e
        print "  Group standard deviation: %.2f" % stddev

# Fit by calling solvePnP() against the other matching pairs and
# averaging the results
for i in range(0,FitIterations):
    ig.fitImagesWithSolvePnP2()
    ig.projectKeypoints(do_grid=True)
    e = ig.groupError(method="average")
    stddev = ig.groupError(method="stddev")
    print "Average error fit, iteration = %d" % i
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

if False:
    ig.placer.placeImages()
    ig.generate_ac3d(c, ref_image=None, base_name="quick-3d", version=0 )
    #s.AffineFitter(steps=30, gain=0.4, fullAffine=False)

if False:
    #image_list = [ "SAM_0031.JPG", "SAM_0032.JPG", "SAM_0033.JPG" ] 
    #image_list = [ "SAM_0153.JPG", "SAM_0154.JPG", "SAM_0155.JPG", "SAM_0156.JPG", "SAM_0157.JPG", "SAM_0158.JPG" ] 
    #image_list = [ "SAM_0184.JPG", "SAM_0185.JPG", "SAM_0186.JPG", "SAM_0187.JPG", "SAM_0188.JPG", "SAM_0189.JPG", "SAM_0190.JPG" ] 
    #image_list = [ "SAM_0199.JPG", "SAM_0200.JPG", "SAM_0201.JPG", "SAM_0202.JPG", "SAM_0203.JPG", "SAM_0204.JPG", "SAM_0205.JPG" ] 
    image_list = [ "SAM_0327.JPG", "SAM_0328.JPG" ]
    print str(image_list)
    #for name in image_list:
    #    image = ig.m.findImageByName(name)
    #    ig.findImageShift(image, gain=1.0, placing=True)
    #    image.placed = True
    ig.render_image_list(image_list, cm_per_pixel=10.0, keypoints=True)

#placed_order = ig.placer.placeImagesByConnections()
group_list = ig.placer.groupByConnections()
placed_list = []
for group in group_list:
    ordered = ig.placer.placeImagesByScore(image_list=group, affine="")
    placed_list = placed_list + ordered
ig.render.drawGrid(placed_list, source_dir=ig.source_dir,
                   cm_per_pixel=10, blend_cm=200, dim=2048 )

#place_list = ig.render.getImagesCoveringPoint(x=.0, y=-40.0, pad=30.0)
#ig.placer.placeImages(place_list)
#ig.render.drawImages(place_list, ig.source_dir,
#                     cm_per_pixel=2.5, blend_cm=200,
#                     keypoints=False)
