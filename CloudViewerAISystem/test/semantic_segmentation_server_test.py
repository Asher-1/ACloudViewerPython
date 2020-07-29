# -*- coding: utf-8 -*-
"""
@author: asher
后台通过接口调用服务，获取OCR识别结果
"""
import os
import sys
import json
import base64
import requests
import numpy as np
from tqdm import tqdm

sys.path.append("..")
from SemanticSegmentationSystem.help_utils import file_processing
from SemanticSegmentationSystem.run_system import InterfaceTest
from SemanticSegmentationSystem.help_utils import cloudviewer_utils as tools


def read_img_base64(p):
    with open(p, 'rb') as f:
        img_string = base64.b64encode(f.read())
    img_string = b'data:image/jpeg;base64,' + img_string
    return img_string.decode()


def post(detection_infos):
    url = 'http://127.0.0.1:9995/aiCloud?'  ##url地址

    http_response = requests.post(url, json=json.dumps(detection_infos))
    data = http_response.content.decode('utf-8')
    res_dict = json.loads(data)
    print("#" * 50)

    if 'state' in res_dict.keys():
        state = res_dict['state']
        print("state: {}".format(state))
    if 'infer_time_take' in res_dict.keys():
        infer_time_take = res_dict['infer_time_take']
        print("infer time take: {} s".format(infer_time_take))
    if 'extraction_time_take' in res_dict.keys():
        extraction_time_take = res_dict['extraction_time_take']
        print("extraction time take: {} s".format(extraction_time_take))

    # write detection result in json format
    with open(os.path.join(TEST_PATH, "result.json"), 'w') as fp:
        fp.write(json.dumps(res_dict, indent=4, ensure_ascii=False))
    print("write detection result to {}".format(os.path.join(TEST_PATH, "result.json")))

    if "instances" in res_dict.keys():
        segmentation_info = res_dict["instances"]
        scene_name_list = list(segmentation_info.keys())
        cloud_list = []
        for scene_name in scene_name_list:
            name_list = scene_name.split(",")
            if len(name_list) == 1:
                cloud_list.append(name_list[0])
            else:
                pc = None
                for name in name_list:
                    file = os.path.join(TEST_PATH, name)
                    if pc is None:
                        pc, _ = tools.IO.read_point_cloud(file)
                    else:
                        pc_array, _ = tools.IO.read_point_cloud(file)
                        pc = np.concatenate((pc, pc_array), axis=0)
                cloud_list.append(pc)
        InterfaceTest.visualize_segmentations(segmentation_info, cloud_list)


if __name__ == '__main__':
    TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test/scene')
    # file_list = file_processing.get_files_list(TEST_PATH, ".pcd")
    # file_list += file_processing.get_files_list(TEST_PATH, ".ply")
    file_list = ["scene2_a.pcd", "scene2_b.pcd", "scene2_c.pcd"]

    info_dict = dict()
    info_dict["files"] = file_list
    steps_dict = dict()
    region_dict = dict()
    region_dict["box"] = [
        {"center": [1.8, -10, 1], "rotation": [0.1, 0.5, 0.6, 1], "extent": [20, 20, 30]},
    ]
    region_dict["sphere"] = [{}, ]
    steps_dict["regions"] = region_dict
    steps_dict["targets"] = {"Utility-Pole": 3, "Insulator": 3}
    info_dict["strategy"] = steps_dict
    # write detection result in json format
    with open(os.path.join(TEST_PATH, "request.json"), 'w') as fp:
        fp.write(json.dumps(info_dict, indent=4, ensure_ascii=False))
    post([info_dict])
