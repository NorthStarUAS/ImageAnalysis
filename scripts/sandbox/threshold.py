#!/usr/bin/python3

# Machine learning (classification) module built with Linear Support
# Vectors

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

class Threshold():

    index = None
    model = None
    cells = {}
    base = None
    saved_labels = []
    saved_data = []
    mode = None                 # LBP, red, ...
    threshold_val = -1
    
    def __init__(self):
        pass

    def init_model(self, rgb, filter):
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if filter == "rgb":
            self.image = rgb.copy()
        elif filter == "hsv":
            self.image = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        elif filter == "hsv90":
            # hsv with hue rotated by a value of 90
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)
            self.hue90 = np.mod((hue.astype('float') + 90), 180).astype('uint8')
            self.image = cv2.merge( (self.hue90, sat, val) )
        else:
            print("Unknown filter:", filter)
            return
        self.min = min
        self.max = max
        
    # and then use the index representation (i.e. LBP) to build the
    # histogram of values
    def gen_score(self, r1, r2, c1, c2):
        mask = self.mask[r1:r2,c1:c2]
        region = self.result[r1:r2,c1:c2]
        dist = 90 - np.abs(90 - self.hue90_masked[r1:r2,c1:c2])
        #cv2.imshow('hue90', dist)
        #cv2.waitKey()
        return (np.max(region), np.average(region), (mask > 0).sum(), np.min(dist))

    # compute the grid layout and classifier
    def compute_grid(self, grid_size=160):
        (h, w) = self.image.shape[:2]
        hcells = int(h / grid_size)
        wcells = int(w / grid_size)
        self.rows = np.linspace(0, h, hcells).astype('int')
        self.cols = np.linspace(0, w, wcells).astype('int')
        self.cells = {}
        for j in range(len(self.rows)-1):
            for i in range(len(self.cols)-1):
                (r1, r2, c1, c2) = (int(self.rows[j]), int(self.rows[j+1]),
                                    int(self.cols[i]), int(self.cols[i+1]))
                key = "%d,%d,%d,%d" % (r1, r2, c1, c2)
                self.cells[key] = { "region": (r1, r2, c1, c2),
                                    "classifier": None,
                                    "user": None,
                                    "prediction": 0,
                                    "score": 0 }
                                    

    def apply_threshold(self, min, max):
        self.mask = cv2.inRange(self.image, min, max)
        self.result = cv2.bitwise_and(self.gray, self.gray, mask=self.mask)
        self.hue90_masked = cv2.bitwise_and(self.hue90, self.hue90, mask=self.mask)
        for key in self.cells:
            (r1, r2, c1, c2) = self.cells[key]["region"]
            self.cells[key]["score"] = self.gen_score(r1, r2, c1, c2)
            #print(key, self.cells[key]["score"])

    # return the key of the cell containing the give x, y pixel coordinate
    def find_key(self, x, y):
        i = np.searchsorted(self.cols, x, side='right') - 1
        j = np.searchsorted(self.rows, y, side='right') - 1
        key = "%d,%d,%d,%d" % (int(self.rows[j]), int(self.rows[j+1]),
                               int(self.cols[i]), int(self.cols[i+1]))
        return key
