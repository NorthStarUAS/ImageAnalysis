# Currently the reference source for this code in this form is the
# 'Curt' branch here:
#
#     https://github.com/clolsonus/GMS-Feature-Matcher/tree/Curt
#
# This version has been modified to better match the naming
# conventions of the original C++ version of the algorithm.  It also
# has some structure copying and float->int conversion fixes so the
# algorithm runs without errors and exactly matches the output of the
# C++ version.
#
# Modifications should be made in the 'Curt' branch at the above link
# and copied here, if 'here' isn't there.
#

import copy
import math
from enum import Enum

import numpy as np
import sys

if sys.version_info[0] < 3:
    print("gms_matcher.py: Must be using Python 3")
    quit()

THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [
    [1, 2, 3,
     4, 5, 6,
     7, 8, 9],

    [4, 1, 2,
     7, 5, 3,
     8, 9, 6],

    [7, 4, 1,
     8, 5, 2,
     9, 6, 3],

    [8, 7, 4,
     9, 5, 1,
     6, 3, 2],

    [9, 8, 7,
     6, 5, 4,
     3, 2, 1],

    [6, 9, 8,
     3, 5, 7,
     2, 1, 4],

    [3, 6, 9,
     2, 5, 8,
     1, 4, 7],

    [2, 3, 6,
     1, 5, 9,
     4, 7, 8]]

# 5 level scales
mScaleRatios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2

class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class GmsMatcher:
    def __init__(self, vuv1, size1, vuv2, size2, vDMatches):
        # Input initialize
        self.mvP1 = self.NormalizePoints(vuv1, size1)
        self.mvP2 = self.NormalizePoints(vuv2, size2)
        self.mNumberMatches = len(vDMatches)
        self.mvMatches = self.ConvertMatches(vDMatches)
        
        # Grid Initialize
        self.mGridSizeLeft = Size(20, 20)
        self.mGridNumberLeft = int(self.mGridSizeLeft.width * self.mGridSizeLeft.height)

        # Initialize the neighbor of left grid
        self.mGridNeighborLeft = np.zeros((self.mGridNumberLeft, 9))
        self.InitializeNeighbors(self.mGridNeighborLeft, self.mGridSizeLeft)

        self.mGridSizeRight = copy.copy(self.mGridSizeLeft)

    # Normalize Key points to range (0-1)
    def NormalizePoints(self, uv_list, size):
        npts = []
        for uv in uv_list:
            npts.append((uv[0] / size.width,
                         uv[1] / size.height))
        return npts

    # Convert OpenCV match to list of tuples
    def ConvertMatches(self, vDMatches):
        vMatches = []
        for match in vDMatches:
            vMatches.append((match.queryIdx, match.trainIdx))
        return vMatches

    def InitializeNeighbors(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def GetInlierMask(self, with_scale, with_rotation):
        max_inlier = 0

        if not with_scale and not with_rotation:
            self.SetScale(0)
            max_inlier = self.run(1)
            return self.mvbInlierMask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.SetScale(scale)
                for RotationType in range(1, 9):
                    num_inlier = self.run(RotationType)
                    print('    ', scale, RotationType, num_inlier)
                    if num_inlier > max_inlier:
                        vb_inliers = self.mvbInlierMask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.mvbInlierMask, max_inlier
        elif with_rotation and not with_scale:
            self.SetScale(0)
            vb_inliers = []
            for RotationType in range(1, 9):
                num_inlier = self.run(RotationType)
                print('    ', RotationType, num_inlier)
                if num_inlier > max_inlier:
                    vb_inliers = self.mvbInlierMask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.mvbInlierMask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.SetScale(scale)
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.mvbInlierMask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.mvbInlierMask, max_inlier

    def SetScale(self, Scale):
        # Set Scale
        self.mGridSizeRight.width = int(self.mGridSizeLeft.width * mScaleRatios[Scale])
        self.mGridSizeRight.height = int(self.mGridSizeLeft.height * mScaleRatios[Scale])
        self.mGridNumberRight = int(self.mGridSizeRight.width * self.mGridSizeRight.height)

        # Initialize the neighbour of right grid
        self.mGridNeighborRight = np.zeros((int(self.mGridNumberRight), 9))
        self.InitializeNeighbors(self.mGridNeighborRight, self.mGridSizeRight)

    def run(self, RotationType):
        self.mvbInlierMask = [False for _ in range(self.mNumberMatches)]

        # Initialize motion statistics
        self.mMotionStatistics = np.zeros((int(self.mGridNumberLeft), int(self.mGridNumberRight)))
        self.mvMatchPairs = [[0, 0] for _ in range(self.mNumberMatches)]

        for GridType in range(1, 5):
            self.mMotionStatistics = np.zeros((int(self.mGridNumberLeft), int(self.mGridNumberRight)))
            self.mCellPairs = [-1 for _ in range(self.mGridNumberLeft)]
            self.mNumberPointsInPerCellLeft = [0 for _ in range(self.mGridNumberLeft)]

            self.AssignMatchPairs(GridType)
            self.VerifyCellPairs(RotationType)

            # Mark inliers
            for i in range(self.mNumberMatches):
                if self.mCellPairs[int(self.mvMatchPairs[i][0])] == self.mvMatchPairs[i][1]:
                    self.mvbInlierMask[i] = True

        return sum(self.mvbInlierMask)

    def AssignMatchPairs(self, grid_type):
        for i in range(self.mNumberMatches):
            lp = self.mvP1[self.mvMatches[i][0]]
            rp = self.mvP2[self.mvMatches[i][1]]
            lgidx = self.mvMatchPairs[i][0] = self.GetGridIndexLeft(lp, grid_type)

            if grid_type == 1:
                rgidx = self.mvMatchPairs[i][1] = self.GetGridIndexRight(rp)
            else:
                rgidx = self.mvMatchPairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.mMotionStatistics[int(lgidx)][int(rgidx)] += 1
            self.mNumberPointsInPerCellLeft[int(lgidx)] += 1

    def GetGridIndexLeft(self, pt, type):
        if type == 1:
            x = pt[0] * self.mGridSizeLeft.width
            y = pt[1] * self.mGridSizeLeft.height
        elif type == 2:
            x = pt[0] * self.mGridSizeLeft.width + 0.5
            y = pt[1] * self.mGridSizeLeft.height
        elif type == 3:
            x = pt[0] * self.mGridSizeLeft.width
            y = pt[1] * self.mGridSizeLeft.height + 0.5
        elif type == 4:
            x = pt[0] * self.mGridSizeLeft.width + 0.5
            y = pt[1] * self.mGridSizeLeft.height + 0.5
        x = int(math.floor(x))
        y = int(math.floor(y))

        if x >= self.mGridSizeLeft.width or y >= self.mGridSizeLeft.height:
            return -1
        return x + y * self.mGridSizeLeft.width

    def GetGridIndexRight(self, pt):
        x = int(math.floor(pt[0] * self.mGridSizeRight.width))
        y = int(math.floor(pt[1] * self.mGridSizeRight.height))
        return x + y * self.mGridSizeRight.width

    def VerifyCellPairs(self, RotationType):
        CurrentRP = ROTATION_PATTERNS[RotationType - 1]

        for i in range(self.mGridNumberLeft):
            if sum(self.mMotionStatistics[i]) == 0:
                self.mCellPairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.mGridNumberRight)):
                value = self.mMotionStatistics[i]
                if value[j] > max_number:
                    self.mCellPairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.mCellPairs[i]
            nb9_lt = self.mGridNeighborLeft[i]
            nb9_rt = self.mGridNeighborRight[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[CurrentRP[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.mMotionStatistics[int(ll), int(rr)]
                thresh += self.mNumberPointsInPerCellLeft[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            
            if score < thresh:
                self.mCellPairs[i] = -2



