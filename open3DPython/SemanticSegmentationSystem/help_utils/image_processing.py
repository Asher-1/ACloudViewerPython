# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 13:05
# @Author  : ludahai
# @FileName: image_processing.py
# @Software: PyCharm

import os
import imageio
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

def accuracy(labels, head_info):
    error_number = 0
    for head in head_info:
        head_name, head_box, head_score = head
        inter_ratio_list = [IOU(head_box_label, head_box) for (head_name_label, head_box_label) in labels]
        if len(inter_ratio_list) == 0:
            continue
        index = np.argmax(inter_ratio_list)
        if inter_ratio_list[index] > 0.5:
            if head_name != labels[index][0]:
                error_number += 1
        else:
            pass
            # error_number += 1
    return error_number


# Deprecated
def helmet_classification(helmet_classifier, image_np, person_head_list, snapshot=False, log_path=None):
    if len(person_head_list) < 1:
        return None, None
    image_np = np.squeeze(image_np)
    im_height = image_np.shape[0]
    im_width = image_np.shape[1]
    margin = 0
    mini_filter_size = helmet_classifier.mini_filter_size
    image_list = []
    new_person_head_list = []
    for person_head in person_head_list:
        box = person_head[0]
        ymin, xmin, ymax, xmax = box
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(xmin * im_width - margin, 0)  # x_min
        bb[1] = np.maximum(ymin * im_height - margin, 0)  # y_min
        bb[2] = np.minimum(xmax * im_width + margin, im_width)  # x_max
        bb[3] = np.minimum(ymax * im_height + margin, im_height)  # y_max
        if bb[2] - bb[0] >= mini_filter_size and bb[3] - bb[1] >= mini_filter_size:
            cropped = image_np[bb[1]:bb[3], bb[0]:bb[2], :]
            image_list.append(cropped)
            new_person_head_list.append(person_head)
    if len(image_list) > 0:
        result_list = helmet_classifier.classify(image_list)
        if snapshot and log_path:
            snapshot_head(image_list, log_path, result_list)
        assert len(image_list) == len(result_list) == len(new_person_head_list)
        return result_list, new_person_head_list
    else:
        return None, None


# Deprecated
def snapshot_head(image_list, log_path, result_list):
    no_helmet_save_dir = os.path.join(log_path, 'no_helmet')
    helmet_save_dir = os.path.join(log_path, 'helmet')
    if not os.path.exists(helmet_save_dir):
        os.makedirs(helmet_save_dir)
    if not os.path.exists(no_helmet_save_dir):
        os.makedirs(no_helmet_save_dir)
    for index, (head_name, head_type_score) in enumerate(result_list):
        if head_name == 'helmet':
            save_dir = helmet_save_dir
        elif head_name == 'no_helmet':
            save_dir = no_helmet_save_dir
        else:
            continue
        try:
            sub_name = strftime("%Y%m%d%H%M%S", gmtime())
            scaled = np.array(Image.fromarray(image_list[index]).resize((256, 256), resample=Image.BILINEAR))
            # scaled = misc.imresize(image_list[index], (256, 256), interp='bilinear')
            imageio.imwrite(os.path.join(save_dir, '{}_{}.{}'.format(str(index), sub_name, 'jpg')), scaled)
        except Exception as e:
            print(e)
