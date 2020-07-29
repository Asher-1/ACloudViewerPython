# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 14:28
# @Author  : ludahai
# @FileName: ai_point_cloud.py
# @Software: PyCharm

import os
import time
import numpy as np
from .configs import cfgs
from .help_utils.logger_utils import logger
from .help_utils.timer_utils.timer_wrapper import timer_wrapper
from .src.object_detector import ObjectDetector

if cfgs.TOOLS_TYPE == "CLOUDVIEWER":
    from .help_utils import cloudviewer_utils as tools
elif cfgs.TOOLS_TYPE == "OPEN3D":
    from .help_utils import open3d_utils as tools


class AIPointCloud(object):
    def __init__(self):
        self.pc_detector = ObjectDetector()
        self.min_points = 100

    @classmethod
    def read_clouds(cls, file, cloud_extent=".ply"):
        # if cloud_extent == '.ply':
        #     pc = tools.IO.read_convert_to_array(file)
        # el
        if cloud_extent == '.xyz' or cloud_extent == '.txt':
            pc_array = tools.IO.load_pc_semantic3d(file, header=None, delim_whitespace=True)
            point_cloud = tools.Utility.array_to_cloud(pc_array)
        else:
            pc_array, point_cloud = tools.IO.read_point_cloud(file)

        return pc_array, point_cloud

    @timer_wrapper
    def semantic_segmentation(self, *data, target_info_list):
        if len(*data) != len(target_info_list):
            message = "target_info dimension must match input data!"
            logger.error(message)
            return {'result': [], 'infer_time_take': 0, 'state': message}

        data_list = []
        targets_infos = []
        json_key_list = []
        has_regions_list = []
        global_map_indices_list = []
        for file_list, target_info_dict in zip(*data, target_info_list):
            if "targets" not in target_info_dict:
                message = "invalid parameters: no targets found!"
                logger.error(message)
                return {'result': [], 'infer_time_take': 0, 'state': message}

            has_regions = "regions" in target_info_dict
            has_regions_list.append(has_regions)
            targets_infos.append(target_info_dict["targets"])
            pc = None
            if len(file_list) > 1:
                json_key_list.append(",".join([os.path.basename(file) for file in file_list]))
                for file in file_list:
                    if has_regions:
                        if pc is None:
                            _, pc = AIPointCloud.read_clouds(file)
                        else:
                            _, pc_obj = AIPointCloud.read_clouds(file)
                            pc += pc_obj
                    else:
                        if pc is None:
                            pc, _ = AIPointCloud.read_clouds(file)
                        else:
                            pc_array, _ = AIPointCloud.read_clouds(file)
                            pc = np.concatenate((pc, pc_array), axis=0)
            elif len(file_list) == 1:
                json_key_list.append(os.path.basename(file_list[0]))
                if has_regions:
                    _, pc = AIPointCloud.read_clouds(file_list[0])
                else:
                    pc, _ = AIPointCloud.read_clouds(file_list[0])
            else:
                message = "cannot find input data!"
                logger.error(message)
                return {'result': [], 'infer_time_take': 0, 'state': message}

            if has_regions:
                cloud = None
                indices = []
                for box_info in target_info_dict["regions"]["box"]:
                    obb = tools.Utility.get_obb_by_params(box_info)
                    inds = obb.get_point_indices_within_bounding_box(pc.get_points())
                    indices.extend(inds)
                    if cloud is None:
                        cloud = pc.select_by_index(inds)
                    else:
                        cloud += pc.select_by_index(inds)
                pc_data = tools.Utility.cloud_to_array(cloud)
                global_map_indices_list.append(np.asarray(indices))
            else:
                pc_data = pc
            data_list.append(pc_data)

        # semantic segmentation
        res = self.pc_detector.semantic_segmentation(data_list)

        if res["state"] == "success":
            pred_list = res["result"]
            start = time.time()
            if len(data_list) == len(pred_list):
                segmentation_instances = self.extract_segmentation(data_list, pred_list,
                                                                   targets_infos, json_key_list)
                for i, scene in enumerate(list(segmentation_instances.keys())):
                    if has_regions_list[i]:
                        for instance in segmentation_instances[scene]:
                            anno_list = segmentation_instances[scene][instance]
                            new_annos = tools.Utility.map_indices(anno_list, global_map_indices_list[i])
                            segmentation_instances[scene][instance] = [anno.tolist() for anno in new_annos]

                res["instances"] = segmentation_instances
            res["extraction_time_take"] = round(time.time() - start, 4)
            res.pop("result")
        else:
            if "result" in list(res.keys()):
                res.pop("result")
        return res

    @timer_wrapper
    def extract_segmentation(self, clouds,
                             predictions,
                             target_info_list,
                             cloud_name_list=None):
        targets_result = dict()
        if len(target_info_list) == 0:
            return targets_result

        cloud_ind = 0
        for pc, seg_pred, target_info in zip(clouds, predictions, target_info_list):
            if isinstance(pc, str):
                pc_array, _ = self.read_clouds(pc)
            else:
                pc_array = pc
            # group point clouds by semantic segmentation labels
            unique_labels, unique_label_indices = \
                tools.Utility.get_unique_label_indices(seg_pred)

            instance_result = {}
            target_list = list(target_info.keys())
            for L, cloud_indices in zip(unique_labels, unique_label_indices):
                # ignore instance whose points number is smaller than 50
                if cloud_indices.shape[0] < self.min_points:
                    logger.info("[Warning] {} tool small number of points, ignore it...".
                                format(cfgs.LABEL_NAME_MAP[L]))
                    continue
                if L == 0:
                    logger.info("ignore {} type".format(cfgs.LABEL_NAME_MAP[L]))
                    continue

                if "All" in target_list:
                    real_index = target_list.index("All")
                elif L in target_list:
                    real_index = target_list.index(L)
                elif cfgs.LABEL_NAME_MAP[L] in target_list:
                    real_index = target_list.index(cfgs.LABEL_NAME_MAP[L])
                else:
                    real_index = -1

                if real_index != -1:
                    pc_obj = tools.Utility.array_to_cloud(pc_array[cloud_indices])
                    point_cloud, reserve_indices = pc_obj.remove_statistical_outlier(30, 1)
                    # compute point cloud_array resolution and get eps
                    eps = point_cloud.compute_resolution() * 30
                    # Cluster PointCloud using the DBSCAN algorithm
                    labels = np.asarray(point_cloud.cluster_dbscan(eps=eps,
                                                                   min_points=self.min_points,
                                                                   print_progress=False))

                    # get cluster indices by given top k value according to points number of each cluster
                    top_k_cluster_indices = tools.Utility.get_clusters_indices_top_k(
                        labels, top_k=target_info[target_list[real_index]], ignore_negative=True)

                    global_indices = tools.Utility.map_indices(top_k_cluster_indices,
                                                               np.asarray(reserve_indices),
                                                               cloud_indices)
                    global_indices = [indices.tolist() for indices in global_indices]
                    if target_list[real_index].lower() == "All":
                        instance_result[cfgs.LABEL_NAME_MAP[L]] = global_indices
                    else:
                        instance_result[target_list[real_index]] = global_indices

            if isinstance(pc, str):
                cloud_key = os.path.basename(pc)
            elif cloud_name_list is not None:
                cloud_key = cloud_name_list[cloud_ind]
            else:
                cloud_key = "scene_{}".format(cloud_ind)

            cloud_ind += 1
            targets_result[cloud_key] = instance_result
        return targets_result
