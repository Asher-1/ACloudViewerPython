# CloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/ReconstructionSystem/register_fragments.py

import numpy as np
import cloudViewer as cv3d
import sys

sys.path.append("../Utility")
from file import join, get_file_list, make_clean_folder
from visualization import draw_registration_result

sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_scene
from refine_registration import multiscale_icp


def preprocess_point_cloud(pcd, config):
    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        cv3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    pcd_fpfh = cv3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        cv3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                              max_nn=100))
    return (pcd_down, pcd_fpfh)


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, config):
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = cv3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            cv3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        result = cv3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True, distance_threshold,
            cv3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 3,
            [
                cv3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnEdgeLength(0.9),
                cv3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            cv3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))
    information = cv3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.get_points()), len(target.get_points())) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)))
    return (True, result.transformation, information)


def compute_initial_registration(s, t, source_down, target_down, source_fpfh,
                                 target_fpfh, path_dataset, config):
    if t == s + 1:  # odometry case
        print("Using RGBD odometry")
        pose_graph_frag = cv3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % s))
        n_nodes = len(pose_graph_frag.nodes)
        transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes -
                                                                  1].pose)
        (transformation, information) = \
            multiscale_icp(source_down, target_down,
                           [config["voxel_size"]], [50], config, transformation_init)
    else:  # loop closure case
        (success, transformation,
         information) = register_point_cloud_fpfh(source_down, target_down,
                                                  source_fpfh, target_fpfh,
                                                  config)
        if not success:
            print("No resonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6, 6)))
    print(transformation)

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)
    return (True, transformation, information)


def update_posegrph_for_scene(s, t, transformation, information, odometry,
                              pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(cv3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            cv3d.pipelines.registration.PoseGraphEdge(s,
                                                      t,
                                                      transformation,
                                                      information,
                                                      uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            cv3d.pipelines.registration.PoseGraphEdge(s,
                                                      t,
                                                      transformation,
                                                      information,
                                                      uncertain=True))
    return (odometry, pose_graph)


def register_point_cloud_pair(ply_file_names, s, t, config):
    print("reading %s ..." % ply_file_names[s])
    source = cv3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = cv3d.io.read_point_cloud(ply_file_names[t])
    (source_down, source_fpfh) = preprocess_point_cloud(source, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target, config)
    (success, transformation, information) = \
        compute_initial_registration(
            s, t, source_down, target_down,
            source_fpfh, target_fpfh, config["path_dataset"], config)
    if t != s + 1 and not success:
        return (False, np.identity(4), np.identity(6))
    if config["debug_mode"]:
        print(transformation)
        print(information)
    return (True, transformation, information)


# other types instead of class?
class matching_result:

    def __init__(self, s, t):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = np.identity(4)
        self.infomation = np.identity(6)


def make_posegraph_for_scene(ply_file_names, config):
    pose_graph = cv3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(cv3d.pipelines.registration.PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t)

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(matching_results), 1))
        results = Parallel(n_jobs=MAX_THREAD)(delayed(
            register_point_cloud_pair)(ply_file_names, matching_results[r].s,
                                       matching_results[r].t, config)
                                              for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]
    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation,
                    matching_results[r].information) = \
                    register_point_cloud_pair(ply_file_names,
                    matching_results[r].s, matching_results[r].t, config)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegrph_for_scene(
                matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation,
                matching_results[r].information, odometry, pose_graph)
    cv3d.io.write_pose_graph(
        join(config["path_dataset"], config["template_global_posegraph"]),
        pose_graph)


def run(config):
    print("register fragments.")
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)
    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")
    make_clean_folder(join(config["path_dataset"], config["folder_scene"]))
    make_posegraph_for_scene(ply_file_names, config)
    optimize_posegraph_for_scene(config["path_dataset"], config)
