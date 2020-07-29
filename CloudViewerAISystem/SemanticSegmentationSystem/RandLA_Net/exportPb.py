# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

from os.path import join, exists
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemantic3D as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os
from tensorflow.python.tools import freeze_graph

MODEL_PATH = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/snapshots/snap-29501'
OUT_DIR = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/pbModels'
PB_NAME = 'semantic3DSegmentation.pb'

DATA_PATH = '/media/yons/data/dataset/pointCloud/data/ownTrainedData'
MODE = 'test'  # 'train' 'None' 'test'
GPU = '1'

TEST_FILE = ""

TEST_MODE = 1

extend = '.xyz'


class Semantic3D:
    def __init__(self):
        self.name = 'Semantic3D'
        self.path = DATA_PATH
        self.label_to_names = {0: 'unlabeled',
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
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        # Following KPConv to do the train-validation split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = 4

        # Initial training-validation-testing files
        self.train_files = []
        self.val_files = []
        self.test_files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == extend]
        for pc_name in cloud_names:
            if exists(join(self.original_folder, pc_name + '.labels')):
                self.train_files.append(join(self.sub_pc_folder, pc_name + '.ply'))
            else:
                self.test_files.append(join(self.full_pc_folder, pc_name + '.ply'))

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)

        for i, file_path in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict for testing
        self.ascii_files = {
            '1_9_local_a.ply': '1_9_local_a_result.labels',
            '1_aa.ply': '1_aa_result.labels',
            '2_aa.ply': '2_aa_result.labels',
            '4_14_local.ply': '4_14_local_result.labels',
            'bildstein_station1_xyz_rgb.ply': 'bildstein_station1_xyz_rgb_result.labels',
            'bildstein_station3_xyz_rgb.ply': 'bildstein_station3_xyz_rgb_result.labels',
            'ckeda_local.ply': 'ckeda_local_result.labels',
            'sg27_station10_rgb.ply': 'sg27_station10_rgb_result.labels',
            'shinan1_9_aa.ply': 'shinan1_9_aa_result.labels',
            'stgallencathedral_station1_rgb.ply': 'stgallencathedral_station1_rgb_result.labels',
        }

        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if file_path in self.val_files:
                cloud_split = 'validation'
            elif file_path in self.train_files:
                cloud_split = 'training'
            else:
                cloud_split = 'test'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        return

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            exist_labels, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            exist_labels = exist_labels.tolist()
            class_total = []
            for ll in self.label_values:
                if ll not in exist_labels:
                    class_total.append(0)
                else:
                    index = exist_labels.index(ll)
                    class_total.append(num_class_total[index])
            num_class_total = np.array(class_total)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
                if split == 'test':
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    queried_pt_weight = 1
                else:
                    queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])
                    # queried_pt_weight = 1

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

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

            for i in range(cfg.num_layers):
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
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
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)


def export_frozenPB():
    tf.reset_default_graph()

    dataset = Semantic3D()

    dataset.init_input_pipeline()

    cfg.saving = False
    model = Network(dataset, cfg)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("we have restred the weights from =====>>\n", MODEL_PATH)
        saver.restore(sess, MODEL_PATH)

        tf.train.write_graph(sess.graph_def, OUT_DIR, PB_NAME)
        freeze_graph.freeze_graph(input_graph=os.path.join(OUT_DIR, PB_NAME),
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=MODEL_PATH,
                                  output_node_names="results/Softmax",
                                  restore_op_name="save/restore_all",
                                  filename_tensor_name='save/Const:0',
                                  output_graph=os.path.join(OUT_DIR, PB_NAME.replace('.pb', '_Frozen.pb')),
                                  clear_devices=False,
                                  initializer_nodes='')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    export_frozenPB()

    # from tensorflow.python import pywrap_tensorflow
    # import os
    #
    # checkpoint_path = os.path.join(MODEL_PATH)
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print('tensor_name: ', key)

