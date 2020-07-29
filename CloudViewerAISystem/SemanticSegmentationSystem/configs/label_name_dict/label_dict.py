# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

class_names = [
    'Unlabeled', 'Manmade-Terrain', 'Natural-Terrain', 'High-Vegetation', 'Low-Vegetation',
    'Buildings', 'Hard-Scape', 'Scanning-Artifacts', 'Cars', 'Utility-Pole', 'Insulator',
    'Electrical-Wire', 'Cross-Bar', 'Stick', 'Fuse', 'Wire-clip', 'Linker-insulator',
    'Persons', 'Traffic-Sign', 'Traffic-Light']

classes_originID = {
    'Unlabeled': 0, 'Manmade-Terrain': 1, 'Natural-Terrain': 2, 'High-Vegetation': 3, 'Low-Vegetation': 4,
    'Buildings': 5, 'Hard-Scape': 6, 'Scanning-Artifacts': 7, 'Cars': 8, 'Utility-Pole': 9,
    'Insulator': 10, 'Electrical-Wire': 11, 'Cross-Bar': 12,
    'Stick': 13, 'Fuse': 14, 'Wire-clip': 15, 'Linker-insulator': 16, 'Persons': 17,
    'Traffic-Sign': 18, 'Traffic-Light': 19}

originID_classes = {item: key for key, item in classes_originID.items()}
NAME_LABEL_MAP = dict(zip(class_names, range(len(class_names))))
LABEL_NAME_MAP = dict(zip(range(len(class_names)), class_names))

# print (originID_classes)



