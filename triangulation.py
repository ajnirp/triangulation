# Rohan Prinja
# COMP 776, Fall 2017
# Assignment: Triangulation and Bundle Adjustment

import numpy as np


#-------------------------------------------------------------------------------

# Given corresponding keypoints in two images, plus a pose matrix P relating the
# two images, obtain a 3D triangulation of the points using a linear method.
#
# inputs:
# - keypoints1: Nx2 array of 2D points in the first image (in normalized camera
#   coordinates)
# - keypoints2: Nx2 array of 2D points in the first image (in normalized camera
#   coordinates)
# - P: camera matrix for mapping 3D points relative to the first image into 3D
#   points relative to the second (P = [R | t])
#
# returns:
# - points3D: Nx3 array of triangulated 3D points
#
def triangulate_points(keypoints1, keypoints2, P):
    points3D = np.empty((len(keypoints1), 3))

    #
    # TODO: Triangulate the correspondences
    #

    assert P.shape == (3, 3)

    return points3D
