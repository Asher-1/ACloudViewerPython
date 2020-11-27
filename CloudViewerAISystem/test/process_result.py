# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 9:58
# @Author  : ludahai
# @FileName: process_result.py
# @Software: PyCharm

import os
import json


def compress(input, output):
    with open(input, 'r', encoding='utf8') as fr:
        info_dict = json.load(fr)
        with open(output, 'w') as fw:
            res = json.dumps(info_dict, ensure_ascii=False)
            fw.write(res)


def decompress(input, output):
    with open(input, 'r', encoding='utf8') as fr:
        info_dict = json.load(fr)
        with open(output, 'w') as fw:
            res = json.dumps(info_dict, indent=2, ensure_ascii=False)
            fw.write(res)


if __name__ == '__main__':
    FILE_PATH = "G:/dataset/pointCloud/data/ownTrainedData/test/whole/"
    input_json = os.path.join(FILE_PATH, "result.json")
    result_json = os.path.join(FILE_PATH, "result_compressed.json")

    compress(input_json, result_json)
    # decompress(input_json, result_json)
