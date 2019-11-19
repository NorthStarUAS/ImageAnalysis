#!/usr/bin/python3

# Machine learning (classification) module built with Linear Support
# Vectors

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import skimage.feature          # pip3 install scikit-image
import sklearn.svm              # pip3 install scikit-learn
#import sklearn.linear_model              # pip3 install scikit-learn

class Classifier():

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

    def init_model(self, basename, threshold_val=-1):
        if threshold_val >= 0:
            self.threshold_val = threshold_val
        else:
            if basename:
                fitname = basename + ".fit"
                dataname = basename + ".data"
            if basename and os.path.isfile(fitname):
                print("Loading SVC model from:", fitname)
                self.model = pickle.load(open(fitname, "rb"))
                #update_prediction(cell_list, model)
            else:
                print("Initializing a new SVC model")
                self.model = sklearn.svm.LinearSVC(max_iter=5000000)
                #self.model = sklearn.linear_model.SGDClassifier(warm_start=True, loss="modified_huber", max_iter=5000000)
            if basename and os.path.isfile(dataname):
                print("Loading saved model data from:", dataname)
                (self.saved_labels, self.saved_data) = \
                    pickle.load( open(dataname, "rb"))
        self.basename = basename

    # compute the Local Binary Pattern representation of the image
    def compute_lbp(self, image, radius=3):
        print("Computing LBP")
        self.mode = "LBP"
        self.radius = radius
        self.numPoints = radius * 8
        smooth = cv2.GaussianBlur(image, (5,5), 4)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        else:
            gray = smooth
        self.index = skimage.feature.local_binary_pattern(gray,
                                                          self.numPoints,
                                                          self.radius,
                                                          method="uniform")

    # compute the "redness" of an image
    def compute_redness(self, rgb):
        print("Computing redness")
        self.mode = "red"
        # very dark pixels can map out noisily
        smooth = cv2.GaussianBlur(rgb, (7,7), 5)
        g, b, r = cv2.split(smooth)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        g[g==0] = 1                 # protect against divide by zero
        ratio = (r / g).astype('float') * 0.25
        # knock out the low end
        lum = gray.astype('float') / 255
        lumf = lum / 0.15
        lumf[lumf>1] = 1
        ratio *= lumf
        ratio[ratio>1] = 1
        self.index = (ratio*255).astype('uint8')
        
    # and then use the index representation (i.e. LBP) to build the
    # histogram of values
    def gen_classifier(self, r1, r2, c1, c2):
        region = self.index[r1:r2,c1:c2]
        if self.threshold_val >= 0:
            return (region >= self.threshold_val).sum()
        elif self.mode == "LBP":
            (hist, _) = np.histogram(region.ravel(),
                                     bins=np.arange(0, self.numPoints + 3),
                                     range=(0, self.numPoints + 2))
        elif self.mode == "red":
            (hist, _) = np.histogram(region.ravel(),
                                     bins=64,
                                     range=(0, 255))
        else:
            print("Unknown mode:", self.mode, "in gen_classifier()")
        hist = hist.astype('float') / region.size # normalize
        if False:
            # dist histogram
            plt.figure()
            y_pos = np.arange(len(hist))
            plt.bar(y_pos, hist, align='center', alpha=0.5)
            plt.xticks(y_pos, range(len(hist)))
            plt.ylabel('count')
            plt.title('classifier')
            plt.show()
        return hist

    # compute the grid layout and classifier
    def compute_grid(self, grid_size=160):
        (h, w) = self.index.shape[:2]
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
                                    
        for key in self.cells:
            (r1, r2, c1, c2) = self.cells[key]["region"]
            self.cells[key]["classifier"] = self.gen_classifier(r1, r2, c1, c2)
            print(key, self.cells[key]["classifier"])

    # do the model fit
    def update_model(self):
        if self.threshold_val >= 0:
            return
        
        labels = list(self.saved_labels)
        data = list(self.saved_data)
        for key in self.cells:
            cell = self.cells[key]
            if cell["user"] != None:
                labels.append(cell["user"])
                data.append(cell["classifier"])
        if len(set(labels)) >= 2:
            print("Updating model fit, training points:", len(data))
            self.model.fit(data, labels)
            if self.basename:
                dataname = self.basename + ".data"
                fitname = self.basename + ".fit"
                print("Saving data:", dataname)
                pickle.dump( (labels, data), open(dataname, "wb"))
                print("Saving model:", fitname)
                pickle.dump(self.model, open(fitname, "wb"))
            print("Done.")
 
    def update_prediction(self):
        if self.threshold_val >= 0:
            for key in self.cells:
                cell = self.cells[key]
                cell["prediction"] = (cell["classifier"] > 0)
                cell["score"] = 1.0
                print(key, cell["prediction"])
            return
        if self.model == None:
            print("No model defined in update_prediction()")
            return
        for key in self.cells:
            cell = self.cells[key]
            if self.threshold_val >= 0:
                cell["prediction"] = (cell["classifier"] > 0)
            else:
                cell["prediction"] = \
                    self.model.predict(cell["classifier"].reshape(1, -1))[0]
                cell["score"] = \
                    self.model.decision_function(cell["classifier"].reshape(1, -1))[0]

    # return the key of the cell containing the give x, y pixel coordinate
    def find_key(self, x, y):
        i = np.searchsorted(self.cols, x, side='right') - 1
        j = np.searchsorted(self.rows, y, side='right') - 1
        key = "%d,%d,%d,%d" % (int(self.rows[j]), int(self.rows[j+1]),
                               int(self.cols[i]), int(self.cols[i+1]))
        return key
