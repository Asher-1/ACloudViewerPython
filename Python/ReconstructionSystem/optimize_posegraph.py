# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/ReconstructionSystem/optimize_posegraph.py

import cloudViewer as cv3d
import sys
sys.path.append("../Utility")
from file import join


def run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
                               max_correspondence_distance,
                               preference_loop_closure):
    # to display messages from cv3d.registration.global_optimization
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)
    method = cv3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = cv3d.registration.GlobalOptimizationConvergenceCriteria()
    option = cv3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=preference_loop_closure,
        reference_node=0)
    pose_graph = cv3d.io.read_pose_graph(pose_graph_name)
    cv3d.registration.global_optimization(pose_graph, method, criteria, option)
    cv3d.io.write_pose_graph(pose_graph_optimized_name, pose_graph)
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Error)


def optimize_posegraph_for_fragment(path_dataset, fragment_id, config):
    pose_graph_name = join(path_dataset,
                           config["template_fragment_posegraph"] % fragment_id)
    pose_graph_optimized_name = join(
        path_dataset,
        config["template_fragment_posegraph_optimized"] % fragment_id)
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
            max_correspondence_distance = config["max_depth_diff"],
            preference_loop_closure = \
            config["preference_loop_closure_odometry"])


def optimize_posegraph_for_scene(path_dataset, config):
    pose_graph_name = join(path_dataset, config["template_global_posegraph"])
    pose_graph_optimized_name = join(
        path_dataset, config["template_global_posegraph_optimized"])
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
            max_correspondence_distance = config["voxel_size"] * 1.4,
            preference_loop_closure = \
            config["preference_loop_closure_registration"])


def optimize_posegraph_for_refined_scene(path_dataset, config):
    pose_graph_name = join(path_dataset, config["template_refined_posegraph"])
    pose_graph_optimized_name = join(
        path_dataset, config["template_refined_posegraph_optimized"])
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
            max_correspondence_distance = config["voxel_size"] * 1.4,
            preference_loop_closure = \
            config["preference_loop_closure_registration"])
