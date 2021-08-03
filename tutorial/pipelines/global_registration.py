# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/global_registration.py

import cloudViewer as cv3d
import numpy as np
import copy
import time


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    cv3d.visualization.draw_geometries([source_temp, target_temp],
                                       zoom=0.4559,
                                       front=[0.6452, -0.3036, -0.7011],
                                       lookat=[1.9892, 2.0208, 1.8945],
                                       up=[-0.2779, -0.9482, 0.1556])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        cv3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = cv3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, cv3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = cv3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    target = cv3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = cv3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        cv3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            cv3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            cv3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], cv3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = cv3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        cv3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = cv3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        cv3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


if __name__ == "__main__":
    voxel_size = 0.05  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down,
                             result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

    voxel_size = 0.05  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size)

    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    print(result_ransac)
    draw_registration_result(source_down, target_down,
                             result_ransac.transformation)

    # Fast global registration
    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)
