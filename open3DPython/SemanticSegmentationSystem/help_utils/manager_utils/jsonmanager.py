#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import json


# attention please:
# the dict must be the same json format
# the dict should not have format the json does not know
# TypeError: Object of type 'datetime' is not JSON serializable

def print_dict_format(data_dict):
    json_str = store_dict_data_to_json_str(data_dict)
    print(json_str)


def store_dict_data_to_json_str(data_dict):
    return json.dumps(data_dict, indent=4, ensure_ascii=False)


def store_dict_data_to_json_file(data_dict, data_json_file_path):
    if os.path.exists(data_json_file_path):
        with open(data_json_file_path, 'w') as data_json_file:
            data_json_file.write(json.dumps(data_dict, indent=4, ensure_ascii=False))


def load_json_str_data_to_dict(data_json_str):
    return json.load(data_json_str)


def load_json_file_data_to_dict(data_json_file_path):
    if os.path.exists(data_json_file_path):
        with open(data_json_file_path) as data_json_file:
            data_dict = json.load(data_json_file)
            return data_dict
