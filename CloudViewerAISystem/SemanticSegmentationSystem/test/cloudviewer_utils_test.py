# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 8:41
# @Author  : ludahai
# @FileName: cloudviewer_utils_test.py
# @Software: PyCharm

import os
import time
import math
import numpy as np
from sklearn.neighbors import KDTree
from SemanticSegmentationSystem.help_utils import file_processing
from SemanticSegmentationSystem.help_utils import cloudviewer_utils as tools
from SemanticSegmentationSystem.help_utils.timer_utils.timer_wrapper import timer_wrapper


@timer_wrapper
def knn_search_1(support_pts, query_pts, k):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """
    assert len(support_pts.shape) == 3
    assert len(query_pts.shape) == 3
    assert support_pts.shape[0] == query_pts.shape[0]
    assert support_pts.shape[2] == query_pts.shape[2]

    batch_number = query_pts.shape[0]
    batch_size = query_pts.shape[1]

    support_batch_size = support_pts.shape[1]
    last_dimension = support_pts.shape[2]
    new_support_pts = np.reshape(support_pts, (batch_number * support_batch_size, last_dimension))
    new_query_pts = np.reshape(query_pts, (batch_number * batch_size, last_dimension))
    kd_tree = tools.KDTree(new_support_pts, leaf_size=50)
    neighbor_idx = kd_tree.query(new_query_pts, k=k, return_distance=False)
    neighbor_idx = neighbor_idx.reshape((batch_number, batch_size, k))

    print("batch computation result shape: {}".format(neighbor_idx.shape))
    return neighbor_idx.astype(np.int32)


@timer_wrapper
def knn_search_2(support_pts, query_pts, k):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """
    assert len(support_pts.shape) == 3
    assert len(query_pts.shape) == 3
    assert support_pts.shape[0] == query_pts.shape[0]
    assert support_pts.shape[2] == query_pts.shape[2]

    batch_number = query_pts.shape[0]
    batch_size = query_pts.shape[1]
    neighbor_idx = np.zeros((batch_number, batch_size, k))

    for iter in range(batch_number):
        kd_tree = tools.KDTree(support_pts[iter], leaf_size=50)
        query_indices = kd_tree.query(query_pts[iter], k=k, return_distance=False)
        neighbor_idx[iter, :, :] = query_indices

    print("one for computation result shape: {}".format(neighbor_idx.shape))
    return neighbor_idx.astype(np.int32)


@timer_wrapper
def knn_search_3(support_pts, query_pts, k):
    assert len(support_pts.shape) == 3
    assert len(query_pts.shape) == 3
    assert support_pts.shape[0] == query_pts.shape[0]
    assert support_pts.shape[2] == query_pts.shape[2]

    batch_number = query_pts.shape[0]
    batch_size = query_pts.shape[1]

    neighbor_idx = np.zeros((batch_number, batch_size, k))

    for iter in range(batch_number):
        kd_tree = tools.KDTree(support_pts[iter], leaf_size=50)
        for i, pts in enumerate(query_pts[iter]):
            idx = kd_tree.query_single(pts, k, return_distance=False)
            neighbor_idx[iter, i, :] = idx

    print("two for computation result shape: {}".format(neighbor_idx.shape))
    return neighbor_idx.astype(np.int32)


def kdtree_test():
    # cloudViewer
    start = time.time()
    kd_tree = tools.KDTree(sub_xyz, leaf_size=50)
    print("cloudViewer construction tree time cost: {}".format(time.time() - start))
    start = time.time()
    # kd_tree = tools.KDTree(tools.Utility.array_to_cloud(sub_xyz), leaf_size=15)
    cv3d_query_idx = kd_tree.query(sub_xyz, k=K, return_distance=False)
    print("cloudViewer query time cost: {}".format(time.time() - start))

    # sklearn
    start = time.time()
    search_tree = KDTree(sub_xyz, leaf_size=50)
    print("sklearn construction tree time cost: {}".format(time.time() - start))
    start = time.time()
    sklearn_query_idx = search_tree.query(sub_xyz, k=K, return_distance=False)
    print("sklearn query time cost: {}".format(time.time() - start))

    print(np.sum(np.sum(sklearn_query_idx - cv3d_query_idx)))


def knn_batch_test():
    pc = sub_xyz[:, :3].astype(np.float32)
    batch_size = math.floor(pc.shape[0] / num_points)
    pc = pc[:batch_size * num_points]
    pc = pc.reshape([batch_size, num_points, 3])
    pc_query = pc[:, :40000, :]

    # nearest neighbours
    res1 = knn_search_1(pc, pc_query, k=K)
    res2 = knn_search_2(pc, pc_query, k=K)
    res3 = knn_search_3(pc, pc_query, k=K)

    diff_1_2 = np.sum(np.sum(np.sum(res1 - res2)))
    diff_2_3 = np.sum(np.sum(np.sum(res2 - res3)))
    diff_1_3 = np.sum(np.sum(np.sum(res3 - res1)))

    print("diff_1_2 {}".format(diff_1_2))
    print("diff_2_3 {}".format(diff_2_3))
    print("diff_3_1 {}".format(diff_1_3))


def boundingbox_test():
    obb_info = {
        "center": [
            1.8,
            -10,
            1
        ],
        "rotation": [
            0.1,
            0.5,
            0.6,
            1
        ],
        "extent": [
            20,
            20,
            30
        ]
    }

    obb = tools.Utility.get_obb_by_params(obb_info)
    indices = obb.get_point_indices_within_bounding_box(point_cloud.get_points())
    pcd = point_cloud.select_by_index(indices)
    tools.Plot.draw_geometries([pcd, obb])


def numpy_test():
    pc = pc_array[:, :3].astype(np.float32)
    pc = pc[:2 * 3, :3]
    reshape_arr = np.reshape(pc, (2, 3, pc.shape[1]))

    arr2 = np.reshape(reshape_arr, (2 * 3, pc.shape[1]))
    print(np.sum(np.sum(arr2 - pc)))

    import perfplot

    def f(x):
        # return math.sqrt(x)
        return np.sqrt(x)

    vf = np.vectorize(f)

    def array_for(x):
        return np.array([f(xi) for xi in x])

    def array_map(x):
        return np.array(list(map(f, x)))

    def fromiter(x):
        return np.fromiter((f(xi) for xi in x), x.dtype)

    def vectorize(x):
        return np.vectorize(f)(x)

    def vectorize_without_init(x):
        return vf(x)

    perfplot.show(
        setup=lambda n: np.random.rand(n),
        n_range=[2 ** k for k in range(20)],
        kernels=[
            f,
            array_for, array_map, fromiter, vectorize, vectorize_without_init
        ],
        logx=True,
        logy=True,
        xlabel='len(x)',
    )


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # os.path.getmtime() last modify time
        # os.path.getctime() last create time
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list


def numpy_test2():
    pc, _ = tools.IO.read_point_cloud(file_list[1])
    print(pc_array.shape)
    print(pc.shape)
    pc_concat = np.concatenate((pc_array, pc,), axis=0)
    print(pc_concat.shape)


if __name__ == '__main__':
    TEST_PATH = os.path.join('G:/dataset/pointCloud/data/ownTrainedData/test')

    file_list = file_processing.get_files_list(TEST_PATH, ".pcd")
    file_list += file_processing.get_files_list(TEST_PATH, ".ply")
    pc_array, point_cloud = tools.IO.read_point_cloud(file_list[0])
    num_points = 81920
    K = 1

    # sub_xyz, sub_colors = tools.Utility.voxel_sampling(
    #     pc_array[:, :3].astype(np.float32), pc_array[:, 3:6], grid_size_scale=1)

    # print(sub_xyz.shape)

    # boundingbox_test()
    # kdtree_test()
    # knn_batch_test()
    # numpy_test()
    # numpy_test2()
