# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 14:04
# @Author  : ludahai
# @FileName: run_system.py
# @Software: PyCharm

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from SemanticSegmentationSystem.configs import cfgs
from SemanticSegmentationSystem.help_utils import file_processing
from SemanticSegmentationSystem.help_utils.logger_utils import logger
from SemanticSegmentationSystem.ai_point_cloud import AIPointCloud
from SemanticSegmentationSystem.src.object_detector import ObjectDetector

if cfgs.TOOLS_TYPE == "CLOUDVIEWER":
    from SemanticSegmentationSystem.help_utils import cloudviewer_utils as tools
elif cfgs.TOOLS_TYPE == "OPEN3D":
    from SemanticSegmentationSystem.help_utils import open3d_utils as tools


def get_segmentation_pair(path, cloud_extent=".ply", label_extent=".labels"):
    cloud_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(path)
                   if os.path.splitext(file_name)[1] == cloud_extent]
    scene_names = []
    label_names = []
    for pc_name in cloud_names:
        if os.path.exists(os.path.join(path, pc_name + cloud_extent)):
            scene_names.append(os.path.join(path, pc_name + cloud_extent))
            label_names.append(os.path.join(path, pc_name + label_extent))
    return scene_names, label_names


def local_test(show_detail=True):
    def semantic_segmentation(label_extent=".labels"):
        pc_list = []
        file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
        for file in file_list:
            pc_array, point_cloud = tools.IO.read_clouds(file)

            ave_color_value = sum(pc_array[:, 3:6]) / pc_array.shape[0]
            if np.average(ave_color_value) < 1:
                pc_array[:, 3:6] = pc_array[:, 3:6] * 255
            pc_list.append(pc_array)

        pc_detector = ObjectDetector()
        res = pc_detector.semantic_segmentation(pc_list)
        if res["state"] == "success":
            pred_list = res["result"]
            for file, pred in zip(file_list, pred_list):
                base, _ = os.path.splitext(file)
                label_file_name = base + label_extent
                np.savetxt(label_file_name, pred, fmt='%d')
                logger.info("Generate labels: {}".format(label_file_name))

    def visualize_results(path, cloud_extent=".ply", label_extent=".labels"):
        scene_names, label_names = get_segmentation_pair(path, cloud_extent, label_extent)
        for scene, label in zip(scene_names, label_names):
            logger.info('scene: {}'.format(scene))
            pc, cloud = tools.IO.read_clouds(scene)
            pc = pc[:, :6].astype(np.float32)
            logger.info('scene point number {}'.format(pc.shape))
            sem_pred = tools.IO.load_label_semantic3d(label)

            # plot
            tools.Plot.draw_pc(pc_xyzrgb=pc[:, 0:6])
            sem_ins_labels = np.unique(sem_pred)
            logger.info('sem_ins_labels: {}'.format(sem_ins_labels))
            tools.Plot.draw_pc_sem_ins(pc_xyz=pc[:, 0:3], pc_sem_ins=sem_pred)

    def extract_instance(path, method="euclidean", cloud_extent=".ply",
                         label_extent=".labels", min_points=300, top_k=10):
        scene_names, label_names = get_segmentation_pair(path, cloud_extent, label_extent)
        cmap = plt.get_cmap("tab20")
        ignore = True
        for scene, label in zip(scene_names, label_names):
            # read point clouds
            pc, _ = tools.IO.read_clouds(scene)
            pc = pc[:, :6].astype(np.float32)
            logger.info('scene point number {}'.format(pc.shape))

            # read semantic segmentation labels
            sem_pred = tools.IO.load_label_semantic3d(label)

            # group point clouds by semantic segmentation labels
            unique_labels, unique_label_indices = tools.Utility.get_unique_label_indices(sem_pred)

            for L, cloud_indices in zip(unique_labels, unique_label_indices):
                # ignore instance whose points number is smaller than 50
                if cloud_indices.shape[0] < MIN_POINTS:
                    print("[Warning] {} tool small number of points, ignore it...".format(cfgs.LABEL_NAME_MAP[L]))
                    continue
                if L == 0:
                    print("ignore {} type".format(cfgs.LABEL_NAME_MAP[L]))
                    continue

                # if cfgs.LABEL_NAME_MAP[L] in ["Utility-Pole", "Insulator", "Electrical-Wire"] or not ignore:
                if cfgs.LABEL_NAME_MAP[L] in \
                        ["Utility-Pole", "Manmade-Terrain", "Insulator", "Electrical-Wire"] or not ignore:
                    if method == "euclidean":
                        seg_instances = tools.Segmentation.euclidean_cluster_segmentation(pc, cloud_indices,
                                                                                          voxel_size=0.02,
                                                                                          min_points=min_points,
                                                                                          top_k=top_k)
                    elif method == "ransac":
                        seg_instances = tools.Segmentation.ransac_segmentation(pc, cloud_indices,
                                                                               classification=cfgs.LABEL_NAME_MAP[L],
                                                                               min_radius=0.001,
                                                                               max_radius=1.0,
                                                                               support_points=300,
                                                                               probability=0.75)
                    else:
                        assert False, "unsupported segmentation method -> {}".format(method)
                    if seg_instances is None:
                        continue

                    labels = np.zeros((pc.shape[0],))
                    labels.fill(-1)
                    # drop ignored clusters
                    cluster_list = tools.Utility.get_clouds_by_indices(pc, seg_instances)
                    i = 1
                    for cloud_cluster, cluster_indices in zip(cluster_list, seg_instances):
                        if show_detail:
                            tools.Plot.draw_pc(cloud_cluster, window_name=cfgs.LABEL_NAME_MAP[L])
                        labels[cluster_indices] = i + 1

                    points_array = pc[cloud_indices]
                    labels = labels[cloud_indices]
                    max_label = labels.max()
                    colors = cmap(labels / (max_label if max_label > 0 else 1))
                    colors[labels < 0] = 0
                    points_array[:, 3:6] = colors[:, :3]
                    tools.Plot.draw_pc(points_array, window_name=cfgs.LABEL_NAME_MAP[L])

    # test segmentation
    # semantic_segmentation(LABEL_EXTENT)9

    # visualization
    visualize_results(TEST_PATH, EXTENT, LABEL_EXTENT)

    # extraction
    extract_instance(TEST_PATH, METHOD, EXTENT, LABEL_EXTENT, MIN_POINTS, TOP_K)


class InterfaceTest(object):
    cloud_ai = AIPointCloud()

    @staticmethod
    def segmentation_test_with_file(show_detail=True):
        file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
        # file_list += file_processing.get_files_list(TEST_PATH, ".ply")
        info_dict = dict()
        info_dict["files"] = file_list
        steps_dict = dict()
        region_dict = dict()
        region_dict["box"] = []
        region_dict["sphere"] = [{}, ]
        steps_dict["regions"] = region_dict
        # steps_dict["targets"] = {"Utility-Pole": 5, "Insulator": 8}
        steps_dict["targets"] = {}
        info_dict["strategy"] = steps_dict
        res = InterfaceTest.cloud_ai.semantic_segmentation([info_dict["files"]],
                                                           target_info_list=[info_dict["strategy"]])
        if res["state"] == "success":
            # write detection result in json format
            tools.IO.write_jsons(os.path.join(TEST_PATH, "result.json"), res, indent=None)
            print("write detection result to {}".format(os.path.join(TEST_PATH, "result.json")))

            segmentation_info = res["instances"]
            scene_name_list = list(segmentation_info.keys())
            cloud_list = InterfaceTest.parse_result(TEST_PATH, scene_name_list)
            InterfaceTest.visualize_segmentations(segmentation_info, cloud_list, show_detail)

    @staticmethod
    def segmentation_test_with_array(show_detail=True):
        file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
        file_list += file_processing.get_files_list(TEST_PATH, ".ply")
        target_info_list = [{"Utility-Pole": 3} for file in file_list]
        cloud_list = []
        for file in file_list:
            pc_array, point_cloud = tools.IO.read_clouds(file)
            cloud_list.append(pc_array)
        res = InterfaceTest.cloud_ai.semantic_segmentation(cloud_list,
                                                           target_info_list=target_info_list)
        if res["state"] == "success":
            # write detection result in json format
            tools.IO.write_jsons(os.path.join(TEST_PATH, "result.json"), res, indent=None)
            print("write detection result to {}".format(os.path.join(TEST_PATH, "result.json")))

            segmentation_info = res["instances"]
            scene_name_list = list(segmentation_info.keys())
            cloud_list = InterfaceTest.parse_result(TEST_PATH, scene_name_list)
            InterfaceTest.visualize_segmentations(segmentation_info, cloud_list, show_detail)

    @staticmethod
    def parse_result(root_path, scene_name_list):
        cloud_list = []
        for scene_name in scene_name_list:
            name_list = scene_name.split(",")
            if len(name_list) == 1:
                cloud_list.append(os.path.join(root_path, name_list[0]))
            else:
                pc = None
                for name in name_list:
                    file = os.path.join(root_path, name)
                    if pc is None:
                        pc, _ = tools.IO.read_clouds(file)
                    else:
                        pc_array, _ = tools.IO.read_clouds(file)
                        pc = np.concatenate((pc, pc_array), axis=0)
                cloud_list.append(pc)
        return cloud_list

    @staticmethod
    def visualization_test(show_detail=True):
        scene_names, label_names = get_segmentation_pair(TEST_PATH, EXTENT, LABEL_EXTENT)
        sem_pred_list = []
        target_info_list = [{"Utility-Pole": 4} for file in scene_names]
        for label_file in label_names:
            # read semantic segmentation labels
            sem_pred = tools.IO.load_label_semantic3d(label_file)
            sem_pred_list.append(sem_pred)
        res = InterfaceTest.cloud_ai.extract_segmentation(scene_names,
                                                          predictions=sem_pred_list,
                                                          target_info_list=target_info_list)

        InterfaceTest.visualize_segmentations(res, scene_names, show_detail)

    @staticmethod
    def visualize_segmentations(segmentation_info, scenes, show_detail=True):
        cmap = plt.get_cmap("tab20")
        for scene, label_key in zip(scenes, list(segmentation_info.keys())):
            # read point clouds
            if isinstance(scene, str):
                pc_array, point_cloud = tools.IO.read_clouds(scene)
            else:
                pc_array = scene
                point_cloud = tools.Utility.array_to_cloud(scene)

            logger.info('scene point number {}'.format(pc_array.shape))
            sem_pred_dict = segmentation_info[label_key]
            labels = np.ndarray([pc_array.shape[0], ])
            labels.fill(-1)
            # drop ignored clusters
            bounding_boxes = []
            for instance_name in list(sem_pred_dict.keys()):
                cluster_list = tools.Utility.get_clouds_by_indices(
                    pc_array, sem_pred_dict[instance_name])
                bboxes = tools.Utility.get_bounding_boxes_by_clouds(cluster_list, [1, 0, 0])
                bounding_boxes += bboxes
                for i, cluster_indices in enumerate(sem_pred_dict[instance_name]):
                    if show_detail:
                        tools.Plot.draw_pc(cluster_list[i], window_name=str(instance_name))
                    labels[cluster_indices] = i + 1

            max_label = labels.max()
            colors = cmap(labels / (max_label if max_label > 0 else 1))[:, :3]
            colors[labels < 0] = pc_array[:, 3:6][labels < 0]
            point_cloud.set_colors(tools.Utility.numpy_to_vector3d(colors))
            tools.Plot.draw_geometries([point_cloud] + bounding_boxes)

    @staticmethod
    def visualize_json_result(json_path, show_detail=True):
        det_dict = tools.IO.read_jsons(json_file=json_path)
        if 'state' in det_dict.keys():
            state = det_dict['state']
            print("state: {}".format(state))
        if 'infer_time_take' in det_dict.keys():
            infer_time_take = det_dict['infer_time_take']
            print("infer time take: {} s".format(infer_time_take))
        if 'extraction_time_take' in det_dict.keys():
            extraction_time_take = det_dict['extraction_time_take']
            print("extraction time take: {} s".format(extraction_time_take))
        if "instances" in det_dict.keys():
            segmentation_info = det_dict["instances"]
            scene_name_list = list(segmentation_info.keys())
            cloud_list = InterfaceTest.parse_result(TEST_PATH, scene_name_list)
            InterfaceTest.visualize_segmentations(segmentation_info, cloud_list, show_detail)


if __name__ == '__main__':
    tools.Utility.set_verbosity_level(level=tools.VerbosityLevel.Debug)
    TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test/whole')
    # TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test/scene')
    JSON_RESULT = os.path.join(TEST_PATH, "result.json")
    # EXTENT = '.xyz'
    EXTENT = ".pcd"
    LABEL_EXTENT = ".labels"
    SHOW_DETAIL = False
    METHOD = "ransac"  # "ransac", "euclidean"

    TOP_K = 100
    MIN_POINTS = 300

    local_test(SHOW_DETAIL)
    # tester = InterfaceTest()
    # tester.segmentation_test_with_file(SHOW_DETAIL)
    # tester.visualization_test(SHOW_DETAIL)
    # tester.visualize_json_result(JSON_RESULT, SHOW_DETAIL)
