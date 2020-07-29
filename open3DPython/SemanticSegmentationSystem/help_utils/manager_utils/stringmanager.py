#!/usr/bin/python
# -*- coding: UTF-8 -*-


import json


def byte_to_str_utf8(src_byte):
    return src_byte.decode("utf-8")


def str_utf8_to_byte(src_str_utf8):
    return src_str_utf8.encode("utf-8")


def dict_str_to_dict(src_dict_str):
    src_dict = json.loads(src_dict_str)
    return src_dict


def dict_to_dict_str(src_dict):
    src_dict_str = json.dumps(src_dict, indent=4, ensure_ascii=False)
    return src_dict_str
