# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 14:04
# @Author  : ludahai
# @FileName: run_system.py
# @Software: PyCharm

import os
import numpy as np
from SemanticSegmentationSystem.configs import cfgs
from SemanticSegmentationSystem.help_utils import file_processing
from SemanticSegmentationSystem.help_utils.logger_utils import logger
from SemanticSegmentationSystem.src.object_detector import ObjectDetector

if cfgs.TOOLS_TYPE == "CLOUDVIEWER":
    from SemanticSegmentationSystem.help_utils import cloudviewer_utils as tools
elif cfgs.TOOLS_TYPE == "OPEN3D":
    from SemanticSegmentationSystem.help_utils import open3d_utils as tools


def semantic_segmentation(label_extent=".labels"):
    pc_list = []
    file_list = file_processing.get_files_list(TEST_PATH, EXTENT)
    for file in file_list:
        if EXTENT == '.ply':
            pc = tools.IO.read_convert_to_array(file)
        elif EXTENT == '.xyz' or EXTENT == '.txt':
            pc = tools.IO.load_pc_semantic3d(file, header=None, delim_whitespace=True)
        else:
            logger.info("unsupported cloud_extent : {}".format(EXTENT))
            continue

        ave_color_value = sum(pc[:, 3:6]) / pc.shape[0]
        if np.average(ave_color_value) < 1:
            pc[:, 3:6] = pc[:, 3:6] * 255
        pc_list.append(pc)

    pc_detector = ObjectDetector()
    res = pc_detector.semantic_segmentation(pc_list)
    if res["state"] == "success":
        preds_list = res["result"]
        for file, preds in zip(file_list, preds_list):
            base, _ = os.path.splitext(file)
            label_file_name = base + label_extent
            np.savetxt(label_file_name, preds, fmt='%d')
            logger.info("Generate labels: {}".format(label_file_name))


def get_segmentation_pair(path, cloud_extent=".ply", label_extent=".labels"):
    cloud_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(path)
                   if os.path.splitext(file_name)[1] == label_extent]
    scene_names = []
    label_names = []
    for pc_name in cloud_names:
        if os.path.exists(os.path.join(path, pc_name + cloud_extent)):
            scene_names.append(os.path.join(path, pc_name + cloud_extent))
            label_names.append(os.path.join(path, pc_name + label_extent))
    return scene_names, label_names


def read_clouds(file, cloud_extent=".ply"):
    # if cloud_extent == '.ply':
    #     pc = tools.IO.read_convert_to_array(file)
    # el
    if cloud_extent == '.xyz' or cloud_extent == '.txt':
        pc = tools.IO.load_pc_semantic3d(file, header=None, delim_whitespace=True)
    else:
        pc = tools.IO.read_point_cloud(file)

    return pc


def visualization(path, cloud_extent=".ply", label_extent=".labels"):
    scene_names, label_names = get_segmentation_pair(path, cloud_extent, label_extent)
    for scene, label in zip(scene_names, label_names):
        logger.info('scene: {}'.format(scene))
        pc = read_clouds(scene)
        pc = pc[:, :6].astype(np.float32)
        logger.info('scene point number {}'.format(pc.shape))
        sem_pred = tools.IO.load_label_semantic3d(label)
        sem_pred.astype(np.float32)

        # plot
        tools.Plot.draw_pc(pc_xyzrgb=pc[:, 0:6])
        sem_ins_labels = np.unique(sem_pred)
        logger.info('sem_ins_labels: {}'.format(sem_ins_labels))
        tools.Plot.draw_pc_sem_ins(pc_xyz=pc[:, 0:3], pc_sem_ins=sem_pred)


def extract_instance(path, cloud_extent=".ply", label_extent=".labels"):
    scene_names, label_names = get_segmentation_pair(path, cloud_extent, label_extent)
    for scene, label in zip(scene_names, label_names):
        pass


if __name__ == '__main__':
    TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test')
    # EXTENT = '.txt'
    EXTENT = '.pcd'
    LABEL_EXTENT = '.lables'

    # test
    # semantic_segmentation(LABEL_EXTENT)

    # visualization
    visualization(TEST_PATH, EXTENT)
