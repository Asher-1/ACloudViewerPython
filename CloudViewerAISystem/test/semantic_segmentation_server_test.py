# -*- coding: utf-8 -*-
"""
@author: asher
"""
import os
import sys
import json
import base64
import requests

sys.path.append("..")
from SemanticSegmentationSystem.run_system import InterfaceTest
from SemanticSegmentationSystem.help_utils import file_processing


def read_img_base64(p):
    with open(p, 'rb') as f:
        img_string = base64.b64encode(f.read())
    img_string = b'data:image/jpeg;base64,' + img_string
    return img_string.decode()


def request_result(detection_infos):
    url = 'http://127.0.0.1:9995/aiCloud?'  ##url地址
    http_response = requests.post(url, json=json.dumps(detection_infos))
    data = http_response.content.decode('utf-8')
    return data


def post(detection_infos):
    data = request_result(detection_infos)
    res_dict = json.loads(data)
    if isinstance(res_dict, str):
        res_dict = json.loads(res_dict)
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
    with open(os.path.join(TEST_PATH, RESULT_FILE_NAME), 'w') as fp:
        fp.write(json.dumps(res_dict, ensure_ascii=False))
    print("write detection result to {}".format(os.path.join(TEST_PATH, "result.json")))

    if "instances" in res_dict.keys():
        segmentation_info = res_dict["instances"]
        scene_name_list = list(segmentation_info.keys())
        cloud_list = InterfaceTest.parse_result(TEST_PATH, scene_name_list)
        InterfaceTest.visualize_segmentations(segmentation_info, cloud_list)


def generate_request_jsons():
    FILE_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test/scene')
    # file_list = file_processing.get_files_list(FILE_PATH, ".pcd")
    # file_list += file_processing.get_files_list(FILE_PATH, ".ply")
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
    steps_dict["targets"] = {"Utility-Pole": 3, "Insulator": 10}
    info_dict["strategy"] = steps_dict
    # write detection result in json format
    with open(os.path.join(FILE_PATH, "request.json"), 'w') as fp:
        fp.write(json.dumps(info_dict, indent=4, ensure_ascii=False))


def inference():
    request_json_path = os.path.join(CUR_PATH, "request_jsons", task)
    with open(request_json_path, 'r', encoding='utf8') as fp:
        info_dict = json.load(fp)
        post([info_dict])


def compress_results(json_file):
    with open(os.path.join(TEST_PATH, json_file), 'r') as fp:
        res_dict = json.load(fp)
        with open(os.path.join(TEST_PATH, "compressed_" + json_file), 'w') as fw:
            json.dump(res_dict, fw, ensure_ascii=False)


if __name__ == '__main__':
    TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test/scene')
    CUR_PATH = os.path.dirname(os.path.abspath(__file__))
    RESULT_FILE_NAME = "result.json"
    task = "full_request.json"

    # inference()
    compress_results(RESULT_FILE_NAME)

    res = request_result([{"file": {"number": 1234}}])
    print(res)
    res2 = json.loads(res)
    print(res2)
    res3 = json.loads(res2)
    print(res3)
