#!/usr/bin/python

import numpy as np
from numpy.linalg import solve, norm
 
 
def ls_lines_intersection(a, n, transpose=True):
    """
    Return the point of intersection computed using the least-square method.
    :param a: a list of point lying on the lines. Points are nx1 matrices
    :param n: a list of line directions. Directions are nx1 matrices.
    :param transpose: should transpose vectors? default true
    :return: the point of intersection
    """
    assert(len(a) == len(n))  # same numbers of points as numbers of directions
 
    if transpose:
        n = list(map(lambda v: np.asmatrix(v/norm(v)).T, n))  # normalize directions and transpose
        a = list(map(lambda v: np.asmatrix(v).T, a))          # transform into matrix and transpose
    else:
        n = list(map(lambda v: np.asmatrix(v/norm(v)).T, n))  # normalize directions
        a = list(map(lambda v: np.asmatrix(v), a))            # transform into matrix

    #print('n:', type(n), n)
    r = np.zeros((n[0].shape[0], n[0].shape[0]))
    q = np.zeros((n[0].shape[0], 1))
    for point, direction in zip(a, n):
        ri = np.identity(direction.shape[0]) - direction.dot(direction.T)
        qi = ri.dot(point)
        r = r + ri
        q = q + qi

    # test
    # x = solve(r, q)
    # q1 = np.dot(r, x)
    # print(x, q1)
    
    return solve(r, q)

# p1 = np.array( [-1, 0, 1] )
# p2 = np.array( [-0.9, 0, 0.9] )
# p3 = np.array( [-1.1, 0.1, 1.1] )
# v1 = np.array( [1, 1, 1] )
# v2 = np.array( [1, -1, 0] )
# v3 = np.array( [0.5, -1, -1] )
# print( ls_lines_intersection([p1, p2, p3], [v1, v2, v3], transpose=True))

# correct answer = [[-1.  ]
#                   [ 0.  ]
#                   [ 0.95]]

