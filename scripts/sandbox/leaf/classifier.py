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
    model = None
    base = None
    fitted = False
    saved_labels = []
    saved_data = []
    
    def __init__(self, basename):
        if basename:
            fitname = basename + ".fit"
            dataname = basename + ".data"
        if basename and os.path.isfile(fitname):
            print("Loading SVC model from:", fitname)
            self.model = pickle.load(open(fitname, "rb"))
            self.fitted = True
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
    def update(self, label_list, classifier_list):
        print("existing data:", len(self.saved_labels))
        labels = list(self.saved_labels)
        data = list(self.saved_data)
        labels += label_list
        data += classifier_list
        print(labels)
        if len(set(labels)) >= 2:
            print("Updating model fit, training points:", len(data))
            self.model.fit(data, labels)
            self.fitted = True
            if self.basename:
                dataname = self.basename + ".data"
                fitname = self.basename + ".fit"
                print("Saving data:", dataname)
                pickle.dump( (labels, data), open(dataname, "wb"))
                print("Saving model:", fitname)
                pickle.dump(self.model, open(fitname, "wb"))
            print("Done.")
 
    def predict(self, label_list, classifier_list):
        if not self.fitted:
            print("Model not yet fit")
            return
        for i in range(len(classifier_list)):
            classifier = np.array(classifier_list[i]).reshape(1, -1)
            # print(classifier)
            result = self.model.predict(classifier)[0]
            label_list[i] = result

    # return the key of the cell containing the give x, y pixel coordinate
    def find_key(self, x, y):
        i = np.searchsorted(self.cols, x, side='right') - 1
        j = np.searchsorted(self.rows, y, side='right') - 1
        key = "%d,%d,%d,%d" % (int(self.rows[j]), int(self.rows[j+1]),
                               int(self.cols[i]), int(self.cols[i+1]))
        return key
