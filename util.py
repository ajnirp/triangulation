# Rohan Prinja
# COMP 776, Fall 2017
# Assignment: Triangulation and Bundle Adjustment

import numpy as np


#-------------------------------------------------------------------------------

# Save a 3D point cloud to a PLY file (viewable in, e.g., MeshLab).
#
# inputs:
# - ply_file: output file
# - points3D: Nx3 array of points
# - color: color string for the points
#
def save_ply(ply_file, points3D, color="255 255 255"):
    with open(ply_file, "w") as fid:
        print>>fid, "ply"
        print>>fid, "format ascii 1.0"
        print>>fid, "element vertex", len(points3D)
        print>>fid, "property float x"
        print>>fid, "property float y"
        print>>fid, "property float z"
        print>>fid, "property uchar red"
        print>>fid, "property uchar green"
        print>>fid, "property uchar blue"
        print>>fid, "end_header"
        for p3D in points3D:
            print>>fid, p3D[0], p3D[1], p3D[2], color


#-------------------------------------------------------------------------------

# Save a camera model mesh
#
# inputs:
# - ply_file: output file
# - P: pose of the camera
# - color: color string for the camera
# - scale: shrink/grow the camera model
#
def save_camera_ply(ply_file, P, color="255 255 255", scale=0.3):
    points3D = scale * np.array((
        (0., 0., 0.),
        (-1., -1., 1.),
        (-1., 1., 1.),
        (1., -1., 1.),
        (1., 1., 1.)))

    # put the points in world coordinates
    RT = P[:3,:3].T
    C = -RT.dot(P[:3,3])
    points3D = points3D.dot(RT) + C

    with open(ply_file, "w") as fid:
        print>>fid, "ply"
        print>>fid, "format ascii 1.0"
        print>>fid, "element vertex", len(points3D)
        print>>fid, "property float x"
        print>>fid, "property float y"
        print>>fid, "property float z"
        print>>fid, "property uchar red"
        print>>fid, "property uchar green"
        print>>fid, "property uchar blue"
        print>>fid, "element face 6"
        print>>fid, "property list uchar int vertex_index"
        print>>fid, "end_header"
        for p3D in points3D:
            print>>fid, p3D[0], p3D[1], p3D[2], color
        print>>fid, "3 0 2 1"
        print>>fid, "3 0 4 2"
        print>>fid, "3 0 3 4"
        print>>fid, "3 0 1 3"
        print>>fid, "3 1 2 4"
        print>>fid, "3 1 4 3"
