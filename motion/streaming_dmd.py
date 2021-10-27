# Source:
#
# https://github.com/cwrowley/dmdtools/blob/master/python/dmdtools/streaming.py
#
# Author: Clancy Rowley
# License: BSD-3-Clause License
#
# Copyright (c) 2015, Clarence Rowley, Maziar Hemati, and Matthew O. Williams
# All rights reserved.

import numpy as np

class StreamingDMD:
    def __init__(self, max_rank=None, ngram=5, epsilon=1.e-10):
        self.max_rank = max_rank
        self.count = 0
        self.ngram = ngram      # number of times to reapply Gram-Schmidt
        self.epsilon = epsilon  # tolerance for expanding the bases

    def update(self, x, y):
        """Update the DMD computation with a pair of snapshots

        Add a pair of snapshots (x,y) to the data ensemble.  Here, if the
        (discrete-time) dynamics are given by z(n+1) = f(z(n)), then (x,y)
        should be measurements corresponding to consecutive states
        z(n) and z(n+1)
        """

        self.count += 1
        normx = np.linalg.norm(x)
        normy = np.linalg.norm(y)
        n = len(x)

        x = np.asmatrix(x).reshape((n, 1))
        y = np.asmatrix(y).reshape((n, 1))

        # process the first iterate
        if self.count == 1:
            # construct bases
            self.Qx = x / normx
            self.Qy = y / normy

            # compute
            self.Gx = np.matrix(normx**2)
            self.Gy = np.matrix(normy**2)
            self.A = np.matrix(normx * normy)
            return

        # ---- Algorithm step 1 ----
        # classical Gram-Schmidt reorthonormalization
        rx = self.Qx.shape[1]
        ry = self.Qy.shape[1]
        xtilde = np.matrix(np.zeros((rx, 1)))
        ytilde = np.matrix(np.zeros((ry, 1)))
        ex = np.matrix(x).reshape((n, 1))
        ey = np.matrix(y).reshape((n, 1))
        for i in range(self.ngram):
            dx = self.Qx.T.dot(ex)
            dy = self.Qy.T.dot(ey)
            xtilde += dx
            ytilde += dy
            ex -= self.Qx.dot(dx)
            ey -= self.Qy.dot(dy)

        # ---- Algorithm step 2 ----
        # check basis for x and expand, if necessary
        if np.linalg.norm(ex) / normx > self.epsilon:
            # update basis for x
            self.Qx = np.bmat([self.Qx, ex / np.linalg.norm(ex)])
            # increase size of Gx and A (by zero-padding)
            self.Gx = np.bmat([[self.Gx, np.zeros((rx, 1))],
                               [np.zeros((1, rx+1))]])
            self.A = np.bmat([self.A, np.zeros((ry, 1))])
            rx += 1

        # check basis for y and expand if necessary
        if np.linalg.norm(ey) / normy > self.epsilon:
            # update basis for y
            self.Qy = np.bmat([self.Qy, ey / np.linalg.norm(ey)])
            # increase size of Gy and A (by zero-padding)
            self.Gy = np.bmat([[self.Gy, np.zeros((ry, 1))],
                               [np.zeros((1, ry+1))]])
            self.A = np.bmat([[self.A],
                              [np.zeros((1, rx))]])
            ry += 1

        # ---- Algorithm step 3 ----
        # check if POD compression is needed
        r0 = self.max_rank
        if r0:
            if rx > r0:
                evals, evecs = np.linalg.eig(self.Gx)
                idx = np.argsort(evals)
                idx = idx[-1:-1-r0:-1]   # indices of largest r0 eigenvalues
                qx = np.asmatrix(evecs[:, idx])
                self.Qx = self.Qx * qx
                self.A = self.A * qx
                self.Gx = np.asmatrix(np.diag(evals[idx]))
            if ry > r0:
                evals, evecs = np.linalg.eig(self.Gy)
                idx = np.argsort(evals)
                idx = idx[-1:-1-r0:-1]   # indices of largest r0 eigenvalues
                qy = np.asmatrix(evecs[:, idx])
                self.Qy = self.Qy * qy
                self.A = qy.T * self.A
                self.Gy = np.asmatrix(np.diag(evals[idx]))

        # ---- Algorithm step 4 ----
        xtilde = self.Qx.T * x
        ytilde = self.Qy.T * y

        # update A and Gx
        self.A += ytilde * xtilde.T
        self.Gx += xtilde * xtilde.T
        self.Gy += ytilde * ytilde.T

    def compute_matrix(self):
        return self.Qx.T.dot(self.Qy).dot(self.A).dot(np.linalg.pinv(self.Gx))

    def compute_modes(self):
        Ktilde = self.compute_matrix()
        evals, evecK = np.linalg.eig(Ktilde)
        modes = self.Qx.dot(evecK)
        return modes, evals
