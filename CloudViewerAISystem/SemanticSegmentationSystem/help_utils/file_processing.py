# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : image_processing.py
    @Author : Asher
    @E-mail : ludahai19@163.com
    @Date   : 2018-12-07 10:10:27
"""

from os import walk
from os import sep
from os import listdir
from os.path import join
from os.path import isdir
from os.path import splitext
from os.path import basename
from os.path import expanduser
import json
import glob
import pickle
import collections
import numpy as np
import configparser
import xml.etree.ElementTree as ET
from collections import OrderedDict


############################################# FOR GENERAL UTILS ##############################################

def write_data(file, content_list, model):
    with open(file, mode=model) as f:
        for line in content_list:
            f.write(basename(line) + "\n")


def read_data(file):
    with open(file, mode="r") as f:
        content_list = f.readlines()
        content_list = [content.rstrip() for content in content_list]
    return content_list


def getFilePathList(file_dir):
    filePath_list = []
    for w in walk(file_dir):
        part_filePath_list = [join(w[0], file) for file in w[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix='ALL'):
    postfix = postfix.split('.')[-1]
    file_list = []
    f_append = file_list.append
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            b_name = basename(file)
            postfix_name = b_name.split('.')[-1]
            if postfix_name == postfix:
                f_append(file)
    file_list.sort()
    return file_list


def gen_files_labels(files_dir, postfix='ALL'):
    # filePath_list = getFilePathList(files_dir)
    filePath_list = get_files_list(files_dir, postfix=postfix)
    print("files nums:{}".format(len(filePath_list)))
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(sep)[-2]
        label_list.append(label)

    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))

    return filePath_list, label_list


class PersonClass(object):
    "Stores the paths to images for a given class"

    def __init__(self, id, name, image_paths):
        self.id = id
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.id + ', ' + self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class GroupClass(object):
    def __init__(self, group, person_classes, image_db=None):
        self.group_name = group
        self.person_set = person_classes
        self.image_db = image_db

    def __str__(self):
        return self.group_name + ', ' + str(len(self.person_set)) + ' person'

    def __len__(self):
        return len(self.person_set)


def get_image_paths(facedir):
    image_paths = []
    if isdir(facedir):
        images = listdir(facedir)
        image_paths = [join(facedir, img) for img in images]
    return image_paths


############################################# FOR FACE RECOGNITION UTILS ##############################################
def getDatasetWithGroups(path, has_class_directories=True):
    dataset = []
    path_exp = expanduser(path)
    groups = [path for path in listdir(path_exp) if isdir(join(path_exp, path))]
    groups.sort()
    nrof_groups = len(groups)
    for i in range(nrof_groups):
        group_name = groups[i]
        id_dir = join(path_exp, group_name)
        classes = [path for path in listdir(id_dir) if isdir(join(id_dir, path))]
        nrof_classes = len(classes)
        persons = []
        for j in range(nrof_classes):
            class_name = classes[j]
            facedir = join(id_dir, class_name)
            image_paths = get_image_paths(facedir)
            name = basename(image_paths[0]).split('_')[1]
            persons.append(PersonClass(class_name, name, image_paths))
        dataset.append(GroupClass(group_name, persons))
    return dataset


def to_json(output_path, *args):
    origin_name = splitext(basename(output_path))[0]
    image_id = "_".join(origin_name.split("_")[-2:])
    with open(output_path, "w") as json_writer:
        boxes, boxes_name, distances, rotate_angle = args
        total_number = len(boxes)
        assert total_number == len(boxes_name) == len(distances)
        result = -1
        data = OrderedDict()
        index = 1
        missing_math_num = 0
        for labels, box, dis in zip(boxes_name, boxes, distances):
            if labels == 'unknown':
                label_list = labels.split('_')
                group = None
                name = None
                user_id = None
                result = 0
                missing_math_num += 1
            else:
                label_list = labels.split('_')
                group = label_list[0]
                name = label_list[1]
                user_id = label_list[2]
                result = -1
            box_with = box[2] - box[0]
            box_height = box[3] - box[1]
            box_left = box[0]
            box_top = box[1]
            probability = min(1 - round(dis, 2) + 0.5, 1)

            face = {"user_id": user_id,
                    "group": group,
                    "probability": str(probability),
                    "result": str(result),
                    "face_rectangle": {
                        "width": str(box_with),
                        "top": str(box_top),
                        "left": str(box_left),
                        "height": str(box_height)
                    }}
            face_key = "{}{}".format("faces", str(index))
            index += 1
            data[face_key] = face
        data["image_id"] = image_id
        data["rotate_angle"] = str(rotate_angle)
        data["total_number"] = str(total_number)
        data["match_number"] = str(total_number - missing_math_num)
        data["match_rate"] = str(round((total_number - missing_math_num) * 1.0 / total_number, 4))
        json_writer.write(json.dumps(data, ensure_ascii=False, sort_keys=False, indent=4))


def convert_to_json(image_id, *args):
    boxes, boxes_name, distances, rotate_angle = args
    total_number = len(boxes)
    assert total_number == len(boxes_name) == len(distances)
    result = -1
    data = OrderedDict()
    index = 1
    missing_math_num = 0
    for labels, box, dis in zip(boxes_name, boxes, distances):
        if labels == 'unknown':
            label_list = labels.split('_')
            group = None
            name = None
            user_id = None
            result = 0
            missing_math_num += 1
        else:
            label_list = labels.split('_')
            group = label_list[0]
            name = label_list[1]
            user_id = label_list[2]
            result = -1
        box_with = box[2] - box[0]
        box_height = box[3] - box[1]
        box_left = box[0]
        box_top = box[1]
        probability = min(1 - round(dis, 2) + 0.5, 1)

        face = {"user_id": user_id,
                "group": group,
                "probability": str(probability),
                "result": str(result),
                "face_rectangle": {
                    "width": str(box_with),
                    "top": str(box_top),
                    "left": str(box_left),
                    "height": str(box_height)
                }}
        face_key = "{}{}".format("faces", str(index))
        index += 1
        data[face_key] = face
    data["image_id"] = image_id
    data["rotate_angle"] = str(rotate_angle)
    data["total_number"] = str(total_number)
    data["match_number"] = str(total_number - missing_math_num)
    data["match_rate"] = str(round((total_number - missing_math_num) * 1.0 / total_number, 4))
    return data


def load_dataset(dataset_path, group_name=None):
    '''
    :param dataset_path: embedding.npy（faceEmbedding.npy）
    :param group_name: the group type
    :return:
    '''

    with open(join(dataset_path, 'labels.pkl'), 'rb') as f:
        label_dict = pickle.load(f)
    names_dict = {}
    if group_name:
        names_dict[group_name] = label_dict[group_name]
    else:
        names_dict = label_dict
    feature_dicts = {}
    for group_labels, names_list in names_dict.items():
        group_name, _ = splitext(basename(group_labels))
        compare_emb = np.load(join(dataset_path, '{}{}'.format(group_name, '.npy')))
        assert len(compare_emb) == len(names_list)
        feature_dicts[group_name] = (names_list, compare_emb)
    return feature_dicts


def log_distances(dis_writer, pic_name, pred_label, distances, box_probs):
    '''
    log distances
    :param dis_writer:
    :param pic_name:
    :param pred_label:
    :param distances:
    :param box_probs:
    :return:
    '''
    assert len(pred_label) == len(distances) == len(box_probs)
    dis_writer.write(pic_name)
    dis_writer.write('-->\t')
    for i in range(len(pred_label)):
        text = '{}({}:{}-{}:{})\t'.format(pred_label[i], 'box_prob',
                                          round(box_probs[i], 4), 'dis', round(distances[i], 2))
        dis_writer.write(text)
    dis_writer.write('\n')


def load_configs(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path)
    secs = conf.sections()
    if len(secs) == 0:
        error_messages = 'cannot find the config file :{}'.format(config_path)
        raise FileNotFoundError(error_messages)
    config_dict = {}
    for sec in secs:
        if sec == 'Section_path':
            ROOT_PATH = conf.get(sec, 'ROOT_PATH')
            for key, value in conf.items(sec):
                config_dict[key.upper()] = join(ROOT_PATH, value)
        else:
            for key, value in conf.items(sec):
                if type(eval(value)) == int:
                    value = int(value)
                elif type(eval(value)) == float:
                    value = float(value)
                elif type(eval(value)) == bool:
                    value = bool(eval(value))
                elif type(eval(value)) == str:
                    value = str(eval(value))
                else:
                    pass
                config_dict[key.upper()] = value
    return config_dict


############################################# FOR HELMET DETECTION UTILS ##############################################
def parse_xml(examples_list):
    """
    parse xml files
    :param examples_list:
    :return: ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    """
    coords = list()
    for xml_file in examples_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objs = root.findall('object')

        for ix, obj in enumerate(objs):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     obj[0].text,
                     int(obj[4][0].text),
                     int(obj[4][1].text),
                     int(obj[4][2].text),
                     int(obj[4][3].text)
                     )
            coords.append(value)
    return coords


def xml_to_csv(xml_path):
    examples_list = glob.glob(join(xml_path, '*.xml'))
    examples = parse_xml(examples_list)
    return examples


def load_labels(xml_path):
    ground_truth_dict = collections.defaultdict(list)
    examples = xml_to_csv(xml_path)
    for example in examples:
        filename, width, height, class_name, xmin, ymin, xmax, ymax = example
        filename = splitext(basename(filename))[0]
        bb = np.zeros(4, dtype=np.int32)
        im_width, im_height = width, height
        bb[0] = np.maximum(xmin, 0)  # x_min
        bb[1] = np.maximum(ymin, 0)  # y_min
        bb[2] = np.minimum(xmax, im_width)  # x_max
        bb[3] = np.minimum(ymax, im_height)  # y_max
        classes_text = class_name
        ground_truth_dict[filename].append((classes_text, bb))

    return ground_truth_dict


def to_json_with_helmet(output_path, *args):
    image_id = splitext(basename(output_path))[0]
    with open(output_path, "w") as json_writer:
        bodys, heads, total_num = args
        danger = 0
        data = OrderedDict()
        index = 1
        no_helmet_num = 0
        missing_math_num = 0
        if len(bodys) == 0:
            missing_math_num = total_num
        for body_info, head_info in zip(bodys, heads):
            result = -1
            body_label, body_box, body_score = body_info
            head_label, head_box, head_score = head_info
            if head_label == 'helmet' or head_label == 'no_helmet':
                if head_label == 'no_helmet':
                    no_helmet_num += 1
                    danger = -1
                result = -1
            else:
                result = 0
                missing_math_num += 1

            body_rectangle_dict = {
                "width": str(body_box[2] - body_box[0]),
                "top": str(body_box[1]),
                "left": str(body_box[0]),
                "height": str(body_box[3] - body_box[1])
            }
            head_rectangle_dict = {
                "width": str(head_box[2] - head_box[0]),
                "top": str(head_box[1]),
                "left": str(head_box[0]),
                "height": str(head_box[3] - head_box[1])
            }
            person = {"body_label": 'body',
                      "body_probability": str(body_score),
                      "body_result": str(result),
                      'body_rectangle': body_rectangle_dict,

                      "head_label": head_label,
                      "head_probability": str(head_score),
                      "head_result": str(result),
                      'head_rectangle': head_rectangle_dict
                      }
            person_key = "{}{}".format("person", str(index))
            index += 1
            data[person_key] = person

        data["image_id"] = image_id
        if total_num == 0:
            data["total_number"] = 0
            data["match_number"] = 0
            data["match_rate"] = 0
        else:
            data["total_number"] = str(total_num)
            data["match_number"] = str(total_num - missing_math_num)
            data["match_rate"] = str(round((total_num - missing_math_num) * 1.0 / total_num, 4))
        data['danger'] = str(danger)
        data['no_helmet_num'] = str(no_helmet_num)
        json_writer.write(json.dumps(data, ensure_ascii=False, sort_keys=False, indent=4))


def convert_to_json_with_helmet(*args):
    image_id, bodys, heads, total_num = args
    danger = 0
    data = OrderedDict()
    no_helmet_num = 0
    for body_info, head_info in zip(bodys, heads):
        head_label, head_box, head_score = head_info
        if head_label == 'no_helmet':
            no_helmet_num += 1
            danger = -1
    data["image_id"] = image_id
    if total_num == 0:
        data["total_number"] = 0
    else:
        data["total_number"] = str(total_num)
    data['danger'] = str(danger)
    data['no_helmet_num'] = str(no_helmet_num)
    return data


def convert_to_json_with_helmet2(*args):
    image_id, bodys, heads, total_num = args
    danger = 0
    data = OrderedDict()
    index = 1
    no_helmet_num = 0
    missing_math_num = 0
    if len(bodys) == 0:
        missing_math_num = total_num
    for body_info, head_info in zip(bodys, heads):
        result = -1
        body_label, body_box, body_score = body_info
        head_label, head_box, head_score = head_info
        if head_label == 'helmet' or head_label == 'no_helmet':
            if head_label == 'no_helmet':
                no_helmet_num += 1
                danger = -1
            result = -1
        else:
            result = 0
            missing_math_num += 1

        body_rectangle_dict = {
            "width": str(body_box[2] - body_box[0]),
            "top": str(body_box[1]),
            "left": str(body_box[0]),
            "height": str(body_box[3] - body_box[1])
        }
        head_rectangle_dict = {
            "width": str(head_box[2] - head_box[0]),
            "top": str(head_box[1]),
            "left": str(head_box[0]),
            "height": str(head_box[3] - head_box[1])
        }
        person = {"body_label": 'body',
                  "body_probability": str(body_score),
                  "body_result": str(result),
                  'body_rectangle': body_rectangle_dict,

                  "head_label": head_label,
                  "head_probability": str(head_score),
                  "head_result": str(result),
                  'head_rectangle': head_rectangle_dict
                  }
        person_key = "{}{}".format("person", str(index))
        index += 1
        data[person_key] = person

    data["image_id"] = image_id
    if total_num == 0:
        data["total_number"] = 0
        data["match_number"] = 0
        data["match_rate"] = 0
    else:
        data["total_number"] = str(total_num)
        data["match_number"] = str(total_num - missing_math_num)
        data["match_rate"] = str(round((total_num - missing_math_num) * 1.0 / total_num, 4))
    data['danger'] = str(danger)
    data['no_helmet_num'] = str(no_helmet_num)
    return data


def wrap_with_dict(image_id, res_list):
    bodys_info, heads_info = res_list
    label_num = len(heads_info)
    return convert_to_json_with_helmet(image_id, bodys_info, heads_info, label_num)


def map_box_to_image(frame, parse_res_list):
    im_height = frame.shape[0]
    im_width = frame.shape[1]
    body_list = []
    head_list = []

    for person_class in parse_res_list:
        for info in [person_class.body_info, person_class.head_info]:
            display_str, box, score = info[:-1]
            ymin, xmin, ymax, xmax = box
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(xmin, 0)  # x_min
            bb[1] = np.maximum(ymin, 0)  # y_min
            bb[2] = np.minimum(xmax, im_width)  # x_max
            bb[3] = np.minimum(ymax, im_height)  # y_max
            if display_str == 'person':
                body_list.append((display_str, bb, score))
            elif display_str == 'helmet' or display_str == 'no_helmet':
                head_list.append((display_str, bb, score))
    return body_list, head_list


if __name__ == '__main__':
    file_dir = 'JPEGImages'
    file_list = get_files_list(file_dir)
    for file in file_list:
        print(file)
