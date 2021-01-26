#!/usr/bin/python3

# Machine learning (classification) module built with Linear Support
# Vectors (adapted for leaves)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sklearn.svm              # pip3 install scikit-learn

class LeafClassifier():
    index = None
    model = None
    cells = {}
    base = None
    saved_labels = []
    saved_data = []
    mode = None                 # LBP, red, ...
    
    def __init__(self, basename):
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

    # do the model fit
    def update_model(self):
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
 
    def predict(self, label_list, classifier_list):
        if self.model == None:
            print("No model defined in update_prediction()")
            return
        for key in self.cells:
            cell = self.cells[key]
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
