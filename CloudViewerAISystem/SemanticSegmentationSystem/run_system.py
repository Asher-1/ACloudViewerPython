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


def read_clouds(file, cloud_extent=".ply"):
    # if cloud_extent == '.ply':
    #     pc = tools.IO.read_convert_to_array(file)
    # el
    if cloud_extent == '.xyz' or cloud_extent == '.txt':
        pc_array = tools.IO.load_pc_semantic3d(file, header=None, delim_whitespace=True)
        point_cloud = tools.Utility.array_to_cloud(pc_array)
    else:
        pc_array, point_cloud = tools.IO.read_point_cloud(file)

    return pc_array, point_cloud


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


def local_test():
    def semantic_segmentation(label_extent=".labels"):
        pc_list = []
        file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
        for file in file_list:
            pc_array, point_cloud = read_clouds(file)

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
            pc, cloud = read_clouds(scene, cloud_extent)
            pc = pc[:, :6].astype(np.float32)
            logger.info('scene point number {}'.format(pc.shape))
            sem_pred = tools.IO.load_label_semantic3d(label)

            # plot
            tools.Plot.draw_pc(pc_xyzrgb=pc[:, 0:6])
            sem_ins_labels = np.unique(sem_pred)
            logger.info('sem_ins_labels: {}'.format(sem_ins_labels))
            tools.Plot.draw_pc_sem_ins(pc_xyz=pc[:, 0:3], pc_sem_ins=sem_pred)

    def extract_instance(path, cloud_extent=".ply", label_extent=".labels", visualization=True):
        scene_names, label_names = get_segmentation_pair(path, cloud_extent, label_extent)
        cmap = plt.get_cmap("tab20")
        for scene, label in zip(scene_names, label_names):
            # read point clouds
            pc, _ = read_clouds(scene)
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

                if cfgs.LABEL_NAME_MAP[L] == "Utility-Pole":
                    # tools.Plot.draw_pc(pc_xyzrgb=cloud_array[:, 0:6], window_name=cfgs.LABEL_NAME_MAP[L])
                    pc_obj = tools.Utility.array_to_cloud(pc[cloud_indices])
                    point_cloud, reserve_indices = pc_obj.remove_statistical_outlier(30, 1)
                    # compute point cloud_array resolution and get eps
                    eps = point_cloud.compute_resolution() * 30
                    logger.info("cluster dbscan eps: {}".format(eps))
                    start = time.time()
                    # Cluster PointCloud using the DBSCAN algorithm
                    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=MIN_POINTS, print_progress=False))
                    print("{} has {} clusters, time cost: {} s".format(
                        cfgs.LABEL_NAME_MAP[L], labels.max() + 1, (time.time() - start)))

                    # get cluster indices by given top k value according to points number of each cluster
                    top_k_cluster_indices = tools.Utility.get_clusters_indices_top_k(
                        labels, top_k=3, ignore_negative=True)

                    full_indices = tools.Utility.map_indices(top_k_cluster_indices,
                                                             np.asarray(reserve_indices),
                                                             cloud_indices)
                    # drop ignored clusters
                    cluster_list = tools.Utility.get_clouds_by_indices(pc, full_indices)

                    if visualization:
                        labels.fill(-1)
                        for i, cluster_indices in enumerate(top_k_cluster_indices):
                            tools.Plot.draw_pc(cluster_list[i])
                            labels[cluster_indices] = i + 1

                        max_label = labels.max()
                        colors = cmap(labels / (max_label if max_label > 0 else 1))
                        colors[labels < 0] = 0
                        point_cloud.set_colors(tools.Utility.numpy_to_vector3d(colors[:, :3]))
                        tools.Plot.draw_geometries([point_cloud])

    # test
    semantic_segmentation(LABEL_EXTENT)

    # visualization
    visualize_results(TEST_PATH, EXTENT, LABEL_EXTENT)

    # extraction
    # extract_instance(TEST_PATH, EXTENT, LABEL_EXTENT)


class InterfaceTest(object):
    cloud_ai = AIPointCloud()

    @staticmethod
    def segmentation_test_with_file():
        file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
        # file_list += file_processing.get_files_list(TEST_PATH, ".ply")
        info_dict = dict()
        info_dict["files"] = file_list
        steps_dict = dict()
        region_dict = dict()
        region_dict["box"] = []
        region_dict["sphere"] = [{}, ]
        steps_dict["regions"] = region_dict
        steps_dict["targets"] = {"Utility-Pole": 3, "Insulator": 3}
        info_dict["strategy"] = steps_dict
        res = InterfaceTest.cloud_ai.semantic_segmentation([info_dict["files"]],
                                                           target_info_list=[info_dict["strategy"]])
        if res["state"] == "success":
            # write detection result in json format
            with open(os.path.join(TEST_PATH, "result.json"), 'w') as fp:
                fp.write(json.dumps(res, indent=4, ensure_ascii=False))
            print("write detection result to {}".format(os.path.join(TEST_PATH, "result.json")))

            segmentation_info = res["instances"]
            scene_name_list = list(segmentation_info.keys())
            cloud_list = InterfaceTest.parse_result(TEST_PATH, scene_name_list)
            InterfaceTest.visualize_segmentations(segmentation_info, cloud_list)

    @staticmethod
    def segmentation_test_with_array():
        file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
        file_list += file_processing.get_files_list(TEST_PATH, ".ply")
        target_info_list = [{"Utility-Pole": 3} for file in file_list]
        cloud_list = []
        for file in file_list:
            pc_array, point_cloud = read_clouds(file)
            cloud_list.append(pc_array)
        res = InterfaceTest.cloud_ai.semantic_segmentation(cloud_list,
                                                           target_info_list=target_info_list)
        if res["state"] == "success":
            # write detection result in json format
            with open(os.path.join(TEST_PATH, "result.json"), 'w') as fp:
                fp.write(json.dumps(res, indent=4, ensure_ascii=False))
            print("write detection result to {}".format(os.path.join(TEST_PATH, "result.json")))

            segmentation_info = res["instances"]
            scene_name_list = list(segmentation_info.keys())
            cloud_list = InterfaceTest.parse_result(TEST_PATH, scene_name_list)
            InterfaceTest.visualize_segmentations(segmentation_info, cloud_list)

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
                        pc, _ = AIPointCloud.read_clouds(file)
                    else:
                        pc_array, _ = AIPointCloud.read_clouds(file)
                        pc = np.concatenate((pc, pc_array), axis=0)
                cloud_list.append(pc)
        return cloud_list

    @staticmethod
    def visualization_test():
        scene_names, label_names = get_segmentation_pair(TEST_PATH, EXTENT, LABEL_EXTENT)
        sem_pred_list = []
        target_info_list = [{"Utility-Pole": 3} for file in scene_names]
        for label_file in label_names:
            # read semantic segmentation labels
            sem_pred = tools.IO.load_label_semantic3d(label_file)
            sem_pred_list.append(sem_pred)
        res = InterfaceTest.cloud_ai.extract_segmentation(scene_names,
                                                          predictions=sem_pred_list,
                                                          target_info_list=target_info_list)

        InterfaceTest.visualize_segmentations(res, scene_names)

    @staticmethod
    def visualize_segmentations(segmentation_info, scenes):
        cmap = plt.get_cmap("tab20")
        for scene, label_key in zip(scenes, list(segmentation_info.keys())):
            # read point clouds
            if isinstance(scene, str):
                pc_array, point_cloud = read_clouds(scene)
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
                    tools.Plot.draw_pc(cluster_list[i], window_name=str(instance_name))
                    labels[cluster_indices] = i + 1

            max_label = labels.max()
            colors = cmap(labels / (max_label if max_label > 0 else 1))[:, :3]
            colors[labels < 0] = pc_array[:, 3:6][labels < 0]
            point_cloud.set_colors(tools.Utility.numpy_to_vector3d(colors))
            tools.Plot.draw_geometries([point_cloud] + bounding_boxes)

    @staticmethod
    def visualize_json_result(json_path):
        with open(json_path, 'r', encoding='utf8')as fp:
            det_dict = json.load(fp)
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
            InterfaceTest.visualize_segmentations(segmentation_info, cloud_list)


if __name__ == '__main__':
    tools.Utility.set_verbosity_level(level=tools.VerbosityLevel.Debug)
    TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test')
    json_result = os.path.join(TEST_PATH, "2_aa_bin_result.json")
    # EXTENT = '.txt'
    EXTENT = ".bin"
    LABEL_EXTENT = ".labels"
    MIN_POINTS = 100

    # local_test()
    tester = InterfaceTest()
    # tester.segmentation_test_with_file()
    # tester.visualization_test()
    tester.visualize_json_result(json_result)
