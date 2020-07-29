#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  RandLA-Net
FILE_NAME    :  DataSet
AUTHOR       :  DAHAI LU
TIME         :  2020/5/11 下午4:11
PRODUCT_NAME :  PyCharm
================================================================
"""

import math
import numpy as np
import tensorflow as tf
import cloudViewer as cv3d
from sklearn.neighbors import KDTree
from .helper_tool import DataProcessing as DP
from .helper_tool import ConfigSemantic3D as cfg

LABEL_MAP = {0: 'unlabeled',
             1: 'man-made-Terrain',
             2: 'natural-Terrain',
             3: 'high-Vegetation',
             4: 'low-Vegetation',
             5: 'buildings',
             6: 'hard-Scape',
             7: 'scanning-Artifacts',
             8: 'cars',
             9: 'utility-Pole',
             10: 'insulator',
             11: 'electrical-Wire',
             12: 'cross-Bar',
             13: 'stick',
             14: 'fuse',
             15: 'wire-clip',
             16: 'linker-insulator',
             17: 'persons',
             18: 'traffic-Sign',
             19: 'traffic-Light'
             }


class DataSetSemantic3D:
    def __init__(self, need_sample=True, leaf_size=50, config=cfg, logger=None, label2Names=LABEL_MAP):
        self.config = config
        self.logger = logger
        self.need_sample = need_sample
        self.leaf_size = leaf_size
        self.name = 'Semantic3D'
        self.cloud_split = "test"
        self.label_to_names = label2Names
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.possibility = {}
        self.min_possibility = {}

        self.raw_size_list = []
        self.test_proj = []
        self.input_trees = {self.cloud_split: []}
        self.input_colors = {self.cloud_split: []}

    def set_input_clouds(self, data_list):
        if self.logger is not None:
            self.logger.info("set input clouds")
        for pc in data_list:
            self.raw_size_list.append(pc.shape[0])
            if self.need_sample:
                if self.config.sampling_method == "cppWrapper":
                    sub_xyz, sub_colors = DP.grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                               pc[:, 3:6].astype(np.uint8),
                                                               grid_size=self.config.sub_grid_size)
                elif self.config.sampling_method == "open3d":
                    sub_xyz, sub_colors = DP.open3d_voxel_sampling(pc[:, :3].astype(np.float32),
                                                                   pc[:, 3:6],
                                                                   grid_size=self.config.sub_grid_size)
                else:
                    self.logger.error("unsupported sampling method : {}".format(self.config.sampling_method))
                    continue

                if self.logger is not None:
                    self.logger.info('down sampling from {} to {}'.
                                     format(self.raw_size_list[-1], sub_xyz.shape[0]))
            else:
                sub_xyz = pc[:, :3].astype(np.float32)
                sub_colors = pc[:, 3:6].astype(np.uint8)

            ave_color_value = sum(sub_colors[0, :]) / sub_colors.shape[1]
            if ave_color_value > 1:
                sub_colors = sub_colors / 255.0

            self.input_colors[self.cloud_split] += [sub_colors]

            if self.logger is not None:
                self.logger.info("kdTree leaf_size : {}".format(self.leaf_size))
            search_tree = KDTree(sub_xyz, leaf_size=self.leaf_size)
            self.input_trees[self.cloud_split] += [search_tree]
            if self.logger is not None:
                self.logger.info("leaf_size: {}".format(self.leaf_size))

            if self.logger is not None:
                self.logger.info("reproject Tree")
            proj_idx = np.squeeze(search_tree.query(pc[:, :3].astype(np.float32), return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            self.test_proj += [proj_idx]

        new_step = math.ceil(sum(self.raw_size_list) * 1.0 / (self.config.num_points * self.config.test_batch_size)) + 1
        if self.logger is not None:
            self.logger.info("update test steps from {} to {}".format(self.config.test_steps, new_step))
            self.logger.info("size list {}".format(self.raw_size_list))
        self.config.test_steps = new_step

    # Generate the input data flow
    def get_batch_gen(self):
        num_per_epoch = self.config.test_steps * self.config.test_batch_size

        # Reset possibility
        self.possibility[self.cloud_split] = []
        self.min_possibility[self.cloud_split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[self.cloud_split]):
            self.possibility[self.cloud_split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[self.cloud_split] += [float(np.min(self.possibility[self.cloud_split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[self.cloud_split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[self.cloud_split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[self.cloud_split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=self.config.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = \
                self.input_trees[self.cloud_split][cloud_idx].query(pick_point, k=self.config.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[self.cloud_split][cloud_idx][query_idx]
                queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                queried_pt_weight = 1

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[self.cloud_split][cloud_idx][query_idx] += delta
                self.min_possibility[self.cloud_split][cloud_idx] = float(
                    np.min(self.possibility[self.cloud_split][cloud_idx]))

                if True:
                    yield (queried_pc_xyz,
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(self.config.num_layers):
                if DP.pool is None:
                    self.config.nn_method = "open3d"

                if self.config.nn_method == "sklearn":
                    neigh_idx = tf.py_func(DP.sklearn_knn_search, [batch_xyz, batch_xyz, self.config.k_n], tf.int32)
                elif self.config.nn_method == "open3d":
                    neigh_idx = tf.py_func(DP.open3d_knn_search, [batch_xyz, batch_xyz, self.config.k_n], tf.int32)
                elif self.config.nn_method == "cppWrapper":
                    neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, self.config.k_n], tf.int32)
                else:
                    assert False

                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // self.config.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // self.config.sub_sampling_ratio[i], :]

                if self.config.nn_method == "sklearn":
                    up_i = tf.py_func(DP.sklearn_knn_search, [sub_points, batch_xyz, 1], tf.int32)
                elif self.config.nn_method == "open3d":
                    up_i = tf.py_func(DP.open3d_knn_search, [sub_points, batch_xyz, 1], tf.int32)
                elif self.config.nn_method == "cppWrapper":
                    up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                else:
                    assert False

                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = tf.concat([transformed_xyz, rgb, features[:, 3:4]], axis=-1)
        return stacked_features

    def init_input_pipeline(self):
        if self.logger is not None:
            self.logger.info('Initiating input pipelines')

        self.config.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]

        if self.logger is not None:
            self.logger.info('ignore label index : {}'.format(self.config.ignored_label_inds))

        gen_function_test, gen_types, gen_shapes = self.get_batch_gen()

        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        self.batch_test_data = self.test_data.batch(self.config.test_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_test_data = self.batch_test_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.prefetch(self.config.test_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_test_data.output_types, self.batch_test_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.test_init_op = iter.make_initializer(self.batch_test_data)
