# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/colored_pointcloud_registration.py

import numpy as np
import copy
import cloudViewer as cv3d


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    cv3d.visualization.draw_geometries([source_temp, target],
                                       zoom=0.5,
                                       front=[-0.2458, -0.8088, 0.5342],
                                       lookat=[1.7745, 2.2305, 0.9787],
                                       up=[0.3109, -0.5878, -0.7468])


if __name__ == "__main__":

    print("1. Load two point clouds and show initial pose")
    source = cv3d.io.read_point_cloud("../../TestData/ColoredICP/frag_115.ply")
    target = cv3d.io.read_point_cloud("../../TestData/ColoredICP/frag_116.ply")

    # draw initial alignment
    current_transformation = np.identity(4)
    draw_registration_result_original_color(source, target, current_transformation)

    # point to plane ICP
    current_transformation = np.identity(4)
    print("2. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    result_icp = cv3d.pipelines.registration.registration_icp(
        source, target, 0.02, current_transformation,
        cv3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(result_icp)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)

    # colored ccPointCloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            cv3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            cv3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = cv3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            cv3d.pipelines.registration.TransformationEstimationForColoredICP(),
            cv3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                               relative_rmse=1e-6,
                                                               max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    draw_registration_result_original_color(source, target, result_icp.transformation)
