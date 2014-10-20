#!/usr/bin/python

import sys

import FlightData
import ImageGroup

class Solver():
    def __init__(self, image_group, correlator):
        self.ig = image_group
        self.c = correlator

    def AffineFitter(self, steps=10, gain=0.5, fullAffine=False):
        for i in xrange(steps):
            self.ig.affineTransformImages(gain=gain, fullAffine=fullAffine)
            self.ig.generate_ac3d(self.c, ref_image=None, base_name="quick-3d", version=i )
            print "Group error (%d) = %.2f: " % (i, self.ig.groupError())

    def SimpleSolver(steps=10, gain=0.25):
        for i in xrange(steps):
            self.ig.rotateImages(gain=gain)
            self.ig.projectKeypoints()
            self.ig.shiftImages(gain=gain)
            self.ig.generate_ac3d(c, ref_image=None, base_name="quick-3d", version=i )
            print "Group error (%d) = %.2f: " % (i, self.ig.groupError())

    # try to globally fit image group by manipulating various
    # parameters and testing to see if that produces a better fit
    # metric
    def estimateParameter(self,
                          param="",
                          min_value=0.0, max_value=1.0, step_size=1.0,
                          refinements=3):
        for i in xrange(refinements):
            best_error = None
            best_value = min_value
            test_value = min_value
            while test_value <= max_value + (step_size*0.1):
                if param == "shutter-latency":
                    self.ig.shutter_latency = test_value
                    self.ig.interpolateAircraftPositions(self.c, force=True)
                elif param == "yaw":
                    self.ig.group_yaw_bias = test_value
                elif param == "roll":
                    self.ig.group_roll_bias = test_value
                elif param == "pitch":
                    self.ig.group_pitch_bias = test_value
                elif param == "altitude":
                    self.ig.group_alt_bias = test_value
                elif param == "k1":
                    self.ig.k1 = test_value
                elif param == "k2":
                    self.ig.k2 = test_value
                self.ig.projectKeypoints()
                error = self.ig.groupError(method="stddev")
                print "Test %s error @ %.5f = %.3f" \
                    % ( param, test_value, error )
                if best_error == None or error < best_error:
                    best_error = error
                    best_value = test_value
                test_value += step_size
            # update values for next iteration
            step_size *= 0.2
            min_value = best_value - step_size*5
            max_value = best_value + step_size*5
        print "Best %s is %.5f (error = %.3f)" % (param, best_value, best_error)
        if param == "shutter-latency":
            self.ig.shutter_latency = best_value
        elif param == "yaw":
            self.ig.group_yaw_bias = best_value
        elif param == "roll":
            self.ig.group_roll_bias = best_value
        elif param == "pitch":
            self.ig.group_pitch_bias = best_value
        elif param == "altitude":
            self.ig.group_alt_bias = best_value
        elif param == "k1":
            self.ig.k1 = best_value
        elif param == "k2":
            self.ig.k2 = best_value
        return best_value


#print "Best shutter latency: " + str(ShutterLatencySolver(0.0, 1.0, 0.1))
#print "Best shutter latency: " + str(ShutterLatencySolver(0.6, 0.8, 0.01))

#bestroll, bestpitch = AttitudeBiasSolver(-5.0, 5.0, 1.0, -5.0, 5.0, 1.0)
#bestroll, bestpitch = AttitudeBiasSolver(-2.0, 0.0, 0.1, -3.0, -1.0, 0.1)
#print "Best roll: %.2f pitch: %.2f" % (bestroll, bestpitch)

#print "Best yaw: " + str(YawBiasSolver(-5, -3, 0.1))
#print "Best altbias: " + str(AltBiasSolver(-8.5, -7.5, 0.1))

#for i in xrange(5):
#    ig.filterStinkers()
#    ig.fitImagesIndividually(gain=0.25)
#    ig.projectKeypoints()
#    print "Group error (after individual fit): %.2f" % ig.groupError()

# AffineSolver(steps=20)
