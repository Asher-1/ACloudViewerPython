# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/color_map_optimization.py

import cloudViewer as cv3d
from trajectory_io import *
import os, sys
sys.path.append("../Utility")
from file import *

# path = "[path_to_fountain_dataset]"
path = "G:/develop/pcl_projects/cloud/TestData/dataset/tutorial/"

debug_mode = False

if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    # Read RGBD images
    rgbd_images = []
    depth_image_path = get_file_list(os.path.join(path, "depth/"), extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"), extension=".png")
    assert (len(depth_image_path) == len(color_image_path))
    for i in range(len(depth_image_path)):
        depth = cv3d.io.read_image(os.path.join(depth_image_path[i]))
        color = cv3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        if debug_mode:
            pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
                rgbd_image,
                cv3d.camera.PinholeCameraIntrinsic(
                    cv3d.camera.PinholeCameraIntrinsicParameters.
                    PrimeSenseDefault))
            cv3d.visualization.draw_geometries([pcd])
        rgbd_images.append(rgbd_image)

    # Read camera pose and mesh
    camera = cv3d.io.read_pinhole_camera_trajectory(os.path.join(path, "scene/trajectory.log"))
    mesh = cv3d.io.read_triangle_mesh(os.path.join(path, "scene", "integrated.ply"))

    # Before full optimization, let's just visualize texture map
    # with given geometry, RGBD images, and camera poses.
    option = cv3d.color_map.ColorMapOptimizationOption()
    option.maximum_iteration = 0
    cv3d.color_map.color_map_optimization(mesh, rgbd_images, camera, option)
    cv3d.visualization.draw_geometries([mesh])
    cv3d.io.write_triangle_mesh(
        os.path.join(path, "scene", "color_map_before_optimization.ply"), mesh)

    # Optimize texture and save the mesh as texture_mapped.ply
    # This is implementation of following paper
    # Q.-Y. Zhou and V. Koltun,
    # Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
    # SIGGRAPH 2014
    option.maximum_iteration = 300
    option.non_rigid_camera_coordinate = True
    cv3d.color_map.color_map_optimization(mesh, rgbd_images, camera, option)
    cv3d.visualization.draw_geometries([mesh])
    cv3d.io.write_triangle_mesh(
        os.path.join(path, "scene", "color_map_after_optimization.ply"), mesh)
