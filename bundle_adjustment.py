# <Your name>
# COMP 776, Fall 2017
# Assignment: Triangulation and Bundle Adjustment

import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from quaternion import rotation_matrix_to_quaternion
from quaternion import quaternion_to_rotation_matrix


#-------------------------------------------------------------------------------

# inputs:
# - keypoints: Nx2 array of 2D points in the first image (in pixels)
# - points3D: Nx3 array of triangulated 3D points
# - R: world-to-camera rotation matrix
# - t: world-to-camera translation vector
# - f: camera focal length
# - cx, cy: camera principal point
# - k1: radial distortion coefficient
#
# returns:
# - reprojection_errors: Nx2 array containing the separate (x, y) distances
#   from each reprojected point to its detected keypoint
#
def calculate_reprojection_errors(keypoints, points3D, R, t, f, cx, cy, k1):
    # apply the camera pose, project
    points3D = points3D.dot(R.T) + t
    points2D = points3D[:,:2] / points3D[:,[2]]

    # apply radial distortion
    r_sq = np.square(points2D).sum(axis=1)
    points2D *= (1. + k1 * r_sq)[:,np.newaxis]

    # put the reprojected points back into pixel coordinates
    points2D *= f
    points2D += (cx, cy)

    reprojection_errors = points2D - keypoints

    return reprojection_errors


#-------------------------------------------------------------------------------

# Given initially triangulated points for a pair of cameras, jointly refine the
# 3D point positions, camera pose (relative rotation+translation of the second
# camera w.r.t the first), and camera intrinsics (here, only focal length and a
# single radial distortion parameter) by minimizing the reprojection error of
# the points into the images
#
# inputs:
# - keypoints1: Nx2 array of 2D points in the first image (in pixels)
# - keypoints2: Nx2 array of 2D points in the first image (in pixels)
# - K: initial camera matrix (shared for both cameras)
# - P: initial pose matrix for the second image w.r.t the first (P = [R | t])
# - points3D: intial Nx3 array of triangulated 3D points
# - k1_init: (optional) initial value for the radial distortion coefficient
#
# returns:
# - f: refined focal length
# - k1: estimated radial distortion coefficient
# - P: refined pose matrix
# - points3D: Nx3 array of refined 3D points
#
def bundle_adjust(keypoints1, keypoints2, K, P, points3D, k1_init = 0.):
    # First, extract relevant parameters.
    f, cx, cy = K[0,0], K[0,2], K[1,2]
    q = rotation_matrix_to_quaternion(P[:,:3])
    t = P[:,3]


    #---------------------------------------------------------------------------
    # Define an objective function; this will return a flattened version of the
    # (x,y) error array E, where the i-th row of E corresponds to the i-th 3D
    # point and has values (e{i}_x1, e{i}_y1, e{i}_x2, e{i}_y2) corresponding to
    # the reprojection errors in the first and second image, respectively.

    def objective(params):
        # convert the flat parameter array into its individual values
        f, k1 = params[:2]
        q = params[2:6]
        t = params[6:9]
        points3D = params[9:].reshape(-1, 3)

        # q and t are both constrained to be unit-length; we'll take a less
        # efficient approach and simply normalize them in the objective
        # note that t should only be constrained to unit length in the two-view
        # case
        R = quaternion_to_rotation_matrix(q / np.linalg.norm(q))
        t = t / np.linalg.norm(t)

        errors = [
            calculate_reprojection_errors(keypoints1, points3D, np.eye(3),
                                          np.zeros(3), f, cx, cy, k1),
            calculate_reprojection_errors(keypoints2, points3D, R, t, f, cx, cy,
                                          k1)]

        return np.column_stack(errors).flatten()


    #---------------------------------------------------------------------------
    # To make solving more efficient, we'll build up a sparse mask capturing
    # which outputs of the loss function are affected by which input parameters.
    #
    # The error vector returned by the objective has 4N elements -- (x,y) errors
    # per image, for each of the N 3D points. The Jacobian of our problem is a
    # (4N)xM matrix J with each row corresponding to an objective function
    # output and each column corresponding to a parameter input. We have 9
    # parameters not related to 3D points (f, k1, q, and t) and 3 parameters
    # (X, Y, and Z) related to each of the 3D points.
    #
    # Each entry of the Jacobian will store the (numerically computed)
    # derivative of the error output w.r.t. the corresponding parameter. Note
    # that J is sparse: The reprojection of each 3D point is only affected by
    # that 3D point's coordinates and the extrinsics/intrinsics. Moreover, only
    # reprojections into the second image are affected by q and t. The Jacobian
    # therefore has the following structure (with 1 representing a non-zero
    # element, and not necessarily the actual value of 1):
    #
    #              f k1 qw qx qy qz tx ty tz X1 Y1 Z1 X2 Y2 Z2 X3 Y3 Z3
    #     e1_x1 [  1  1  0  0  0  0  0  0  0  1  1  1  0  0  0  0  0  0       ]
    #     e1_y1 [  1  1  0  0  0  0  0  0  0  1  1  1  0  0  0  0  0  0       ]
    #     e1_x2 [  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0       ]
    #     e1_y2 [  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0       ]
    #     e2_x1 [  1  1  0  0  0  0  0  0  0  0  0  0  1  1  1  0  0  0       ]
    #     e2_y1 [  1  1  0  0  0  0  0  0  0  0  0  0  1  1  1  0  0  0       ]
    # J = e2_x2 [  1  1  1  1  1  1  1  1  1  0  0  0  1  1  1  0  0  0 . . . ]
    #     e2_y2 [  1  1  1  1  1  1  1  1  1  0  0  0  1  1  1  0  0  0       ]
    #     e3_x1 [  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1       ]
    #     e3_y1 [  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1       ]
    #     e3_x2 [  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0  1  1  1       ]
    #     e3_y2 [  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0  1  1  1       ]
    #           [                       .  .  .                               ]

    N = len(points3D)
    jacobian_mask = lil_matrix((N * 4, 9 + points3D.size), dtype=np.bool)
    jacobian_mask[:, :2] = True     # all points are affected by f and k1
    jacobian_mask[2::4, 2:9] = True # (q, t) apply to second image only
    jacobian_mask[3::4, 2:9] = True # (q, t) apply to second image only
    for i in xrange(len(points3D)):
        r, c = 4 * i, 3 * i + 9
        jacobian_mask[r:r+4, c:c+3] = True
    jacobian_mask = jacobian_mask.tobsr()


    #---------------------------------------------------------------------------
    # run the optimization, solve using scipy's least_squares function
    
    print "Starting Bundle Adjustment"
    init_params = np.concatenate(([f, k1_init], q, t, points3D.flatten()))
    res = least_squares(objective, init_params, jac_sparsity=jacobian_mask,
        verbose=2)


    #---------------------------------------------------------------------------
    # get the optimization result and return

    params = res.x
    f, k1 = params[:2]
    q = params[2:6]
    t = params[6:9]
    points3D = params[9:].reshape(-1, 3)

    R = quaternion_to_rotation_matrix(q / np.linalg.norm(q))
    t = t / np.linalg.norm(t)

    P = np.column_stack((R, t))

    return f, k1, P, points3D
