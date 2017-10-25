# Rohan Prinja
# COMP 776, Fall 2017
# Assignment: Triangulation and Bundle Adjustment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

import os
import numpy as np

from bundle_adjustment import bundle_adjust
from triangulation import triangulate_points
from util import save_ply, save_camera_ply


#-------------------------------------------------------------------------------

def main(args):
    #---------------------------------------------------------------------------
    # load camera intrinsic matrix; note that in general, we will have a
    # different K matrix for each image, but we assume that the parameters are
    # shared, here (i.e., the two images are taken with the same camera)

    K = np.loadtxt(args.K)

    #---------------------------------------------------------------------------
    # load corresponding keypoints and normalize them

    keypoints1 = np.loadtxt(args.keypoints1)
    keypoints2 = np.loadtxt(args.keypoints2)

    keypoints1_normalized = (keypoints1 - K[:2,2]) / (K[0,0], K[1,1])
    keypoints2_normalized = (keypoints2 - K[:2,2]) / (K[0,0], K[1,1])


    #---------------------------------------------------------------------------
    # load F, compute 4 possible solutions for R and t
    # the sign checking on U and VT is necessary to get R matrices with det R=1

    F = np.loadtxt(args.F)
    E = K.T.dot(F).dot(K)
    U, S, VT = np.linalg.svd(E)
    if np.isclose(np.linalg.det(U), -1.):
        U *= -1. 
    if np.isclose(np.linalg.det(VT), -1.):
        VT *= -1. 
    W = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))


    #---------------------------------------------------------------------------
    # triangulate 3D points X for each of the 4 possible decompositions

    P1 = np.column_stack((U.dot(W).dot(VT), U[:,2]))
    X1 = triangulate_points(keypoints1_normalized, keypoints2_normalized, P1)

    P2 = np.column_stack((U.dot(W.T).dot(VT), U[:,2]))
    X2 = triangulate_points(keypoints1_normalized, keypoints2_normalized, P2)

    P3 = np.column_stack((U.dot(W).dot(VT), -U[:,2]))
    X3 = triangulate_points(keypoints1_normalized, keypoints2_normalized, P3)

    P4 = np.column_stack((U.dot(W.T).dot(VT), -U[:,2]))
    X4 = triangulate_points(keypoints1_normalized, keypoints2_normalized, P4)


    #---------------------------------------------------------------------------
    # choose the best decomposition as the one having the most points in front
    # of the cameras

    P, points3D = max(
        (P1, X1), (P2, X2), (P3, X3), (P4, X4),
        key=lambda e: np.count_nonzero(e[1][:, 2] > 0))

    # remove points that failed to be triangulated, if any
    mask = (points3D[:, 2] > 0)
    points3D = points3D[mask]
    keypoints1 = keypoints1[mask]
    keypoints2 = keypoints2[mask]


    #---------------------------------------------------------------------------
    # save the initial results to PLY files

    points3D_orig = points3D.copy()
    P_orig = P.copy()

    print
    print "Original P Matrix:"
    print P_orig
    print

    # save the initial result in cyan
    save_ply(os.path.join(args.output_path, "triangulated_points.ply"),
             points3D_orig, "0 255 255")

    # the first camera has identity pose; save it in gray 
    save_camera_ply(os.path.join(args.output_path, "camera1.ply"),
                    np.column_stack((np.eye(3), np.zeros(3))), "128 128 128")

    # save the original second camera in cyan
    save_camera_ply(os.path.join(args.output_path, "camera2.ply"), P_orig,
                    "0 255 255")


    #---------------------------------------------------------------------------
    # run bundle adjustment to jointly refine the 3D point positions, camera
    # pose (relative rotation+translation of the second camera w.r.t the first),
    # and camera intrinsics (here, only focal length and a single radial
    # distortion parameter)
    
    f, k1, P, points3D = bundle_adjust(
        keypoints1, keypoints2, K, P, points3D)

    print "Refined focal length:", f
    print "Estimated radial distortion coefficient:", k1
    print "Refined P Matrix:"
    print P


    #---------------------------------------------------------------------------
    # save the bundle adjusted results to PLY files

    # save the bundle adjustment result in yellow
    save_ply(os.path.join(args.output_path,
                "triangulated_points_bundle_adjusted.ply"),
             points3D, "255 255 0")

    # save the bundle-adjusted second camera in yellow
    save_camera_ply(os.path.join(args.output_path,
                        "camera2_bundle_adjusted.ply"),
                    P, "255 255 0")


    #---------------------------------------------------------------------------
    # display the result in matplotlib (very slow 3D visualization, sadly)

    #fig = plt.figure()
    #ax = plt.subplot(111, projection="3d")
    #ax.scatter(
    #    points3D_orig[:, 0], points3D_orig[:, 1], points3D_orig[:, 2], c="r")
    #ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c="g")
    #
    ## exit when a key is pressed
    #while not plt.waitforbuttonpress(): pass


#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Given matched points in two images taken with the same "
            "camera, a precomputed fundamental matrix relating the images, and "
            "the camera's intrinsic matrix, perform linear 3D triangulation of "
            "the points and run a bundle adjustment on the result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--K", type=str, default="K.txt",
        help="camera intrinsic matrix (assumed to be the same for both images)")

    parser.add_argument("--F", type=str, default="F.txt",
        help="precomputed fundamental matrix for the image pair s.t. "
            "x_2^T F x_1 = 0")

    parser.add_argument("--keypoints1", type=str, default="keypoints1.txt",
        help="keypoints for the first image, with each line containing the "
            "corresponding point for the same line in keypoints2")

    parser.add_argument("--keypoints2", type=str, default="keypoints2.txt",
        help="keypoints for the second image, with each line containing the "
            "corresponding point for the same line in keypoints1")

    parser.add_argument("--output_path", type=str, default=".")

    args = parser.parse_args()

    main(args)
