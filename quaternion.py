# <Your name>
# COMP 776, Fall 2017
# Assignment: Triangulation and Bundle Adjustment

import numpy as np

#-------------------------------------------------------------------------------

# R: 3x3 rotation matrix, assumed to be valid
# returns: unit-length quaternion corresponding to R
def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)

    if trace > 0:
        qw = 0.5 * np.sqrt(1. + trace)
        qx = (R[2,1] - R[1,2]) * 0.25 / qw
        qy = (R[0,2] - R[2,0]) * 0.25 / qw
        qz = (R[1,0] - R[0,1]) * 0.25 / qw
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2. * np.sqrt(1. + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2. * np.sqrt(1. + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s = 2. * np.sqrt(1. + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s

    return np.array((qw, qx, qy, qz))


#-------------------------------------------------------------------------------

# q: quaternion repesented as a 4-element numpy array, assumed to be unit length
# returns: 3x3 rotation matrix corresponding to q
def quaternion_to_rotation_matrix(q):
    return np.eye(3) + 2 * np.array((
      (-q[2] * q[2] - q[3] * q[3],
        q[1] * q[2] - q[3] * q[0],
        q[1] * q[3] + q[2] * q[0]),
      ( q[1] * q[2] + q[3] * q[0],
       -q[1] * q[1] - q[3] * q[3],
        q[2] * q[3] - q[1] * q[0]),
      ( q[1] * q[3] - q[2] * q[0],
        q[2] * q[3] + q[1] * q[0],
       -q[1] * q[1] - q[2] * q[2])))
