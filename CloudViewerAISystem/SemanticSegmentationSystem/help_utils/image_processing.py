# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 13:05
# @Author  : ludahai
# @FileName: image_processing.py
# @Software: PyCharm

import os
import numpy as np
from PIL import Image
from time import gmtime, strftime


class PersonClass(object):
    "Stores the paths to images for a given class"

    def __init__(self, body_info, head_info):
        assert isinstance(body_info, tuple) and len(body_info) == 4
        assert isinstance(head_info, tuple) and len(head_info) == 4
        self.body_info = body_info
        self.head_info = head_info
        self.__body_name = self.body_info[0]
        self.__body_box = self.body_info[1]
        self.__body_score = self.body_info[2]
        self.__body_color = self.body_info[3]
        self.__head_name = self.head_info[0]
        self.__head_box = self.head_info[1]
        self.__head_score = self.head_info[2]
        self.__head_color = self.head_info[3]

    @property
    def body_name(self):
        return self.__body_name

    @property
    def body_box(self):
        return self.__body_box

    @property
    def body_score(self):
        return self.__body_score

    @property
    def body_color(self):
        return self.__body_color

    @property
    def head_name(self):
        return self.__head_name

    @property
    def head_box(self):
        return self.__head_box

    @property
    def head_score(self):
        return self.__head_score

    @property
    def head_color(self):
        return self.__head_color

    def __str__(self):
        text = self.body_name + ', ' + str(self.body_score) + ', ' + self.body_color + ', '
        text += self.head_name + ', ' + str(self.head_score) + ', ' + self.head_color
        return text


def IOU(box1, box2):
    """
    :param box1:[x1,y1,x2,y2] the left corner and the right corner
    :param box2:[x1,y1,x2,y2]
    :return: iou_ratio intersection ratio
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3])
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0], box1[2], box2[0], box2[2])
    y_max = max(box1[1], box1[3], box2[1], box2[3])
    x_min = min(box1[0], box1[2], box2[0], box2[2])
    y_min = min(box1[1], box1[3], box2[1], box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area)
    return iou_ratio


def intersection_ratio(box1, box2):
    """
    :param box1: person box
    :param box2: head box
    :return:
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3])
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0], box1[2], box2[0], box2[2])
    y_max = max(box1[1], box1[3], box2[1], box2[3])
    x_min = min(box1[0], box1[2], box2[0], box2[2])
    y_min = min(box1[1], box1[3], box2[1], box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        return 0
    iou_area = iou_width * iou_height
    return iou_area * 1.0 / (width2 * height2)
