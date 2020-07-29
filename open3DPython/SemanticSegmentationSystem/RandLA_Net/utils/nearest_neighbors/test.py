import os
import time
import math
import numpy as np
import open3d as o3d
from helper_ply import read_ply
from helper_tool import DataProcessing as DP
import lib.python.nearest_neighbors as nearest_neighbors
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from manager_utils.sklearn_neighbors import SklearnNeighbors
import multiprocessing as mp


def read_convert_to_array(file):
    pc = read_ply(file)
    pc_xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T
    pc_colors = np.vstack((pc['red'], pc['green'], pc['blue'])).T
    return np.hstack([pc_xyz, pc_colors])


def open3d_knn_search(support_pts, query_pts, k):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """

    assert len(support_pts.shape) == 3
    assert len(query_pts.shape) == 3

    batch_number = query_pts.shape[0]
    batch_size = query_pts.shape[1]

    data = np.swapaxes(np.vstack(support_pts), 0, 1)
    kd_tree = o3d.geometry.KDTreeFlann(data)
    neighbor_idx = np.zeros((batch_number, batch_size, k))
    # query_data = np.vstack(query_pts)
    for iter in range(batch_number):
        for i, pts in enumerate(query_pts[iter]):
            [dummy, idx, _] = kd_tree.search_knn_vector_3d(pts, k)
            neighbor_idx[iter, i, :] = np.asarray(idx)
            # neighbor_idx.append(np.asarray(idx))
    # return np.array(neighbor_idx).astype(np.int32).reshape((batch_number, batch_size, k))
    return neighbor_idx.astype(np.int32)


def demo0():
    tree = KDTree(pc, leaf_size=40)
    dummy, indices = tree.query(pc, k=K)
    print(dummy)
    print(indices)


def demo3():
    start = time.time()
    pcd = o3d.io.read_point_cloud("D:/develop/workstations/PCL_projects/cloud/test/moduleCloud.pcd")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    print("Find its 200 nearest neighbors, paint blue.")
    print(pcd.points)
    arr = np.asarray(pcd.points)
    arr = np.swapaxes(arr, 0, 1)
    pcd_tree = o3d.geometry.KDTreeFlann(arr)
    print(arr)
    print(type(pcd.points))
    query = np.stack(pcd.points[:1])
    query = query.reshape([query.size, 1])
    query2 = np.swapaxes(pcd.points[:2], 0, 1)
    query2 = query2.reshape([query2.size, 1])
    [k, idx, _] = pcd_tree.search_knn_vector_3d(query, 200)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

    print("Visualize the point cloud.")
    o3d.visualization.draw_geometries([pcd])
    print(idx)
    print(time.time() - start)


def demo4():
    file_list = DP.get_files_list(TEST_PATH, extend)
    file = file_list[0]
    if extend == '.ply':
        pc = read_convert_to_array(file)
    elif extend == '.xyz':
        pc = DP.load_pc_semantic3d(file, header=None, delim_whitespace=True)
    else:
        exit(0)

    pc = pc[:, :3].astype(np.float32)
    batch_size = math.floor(pc.shape[0] / num_points)
    pc = pc[:batch_size * num_points]
    pc = pc.reshape([batch_size, num_points, 3])
    pc_query = pc[:, :3, :]

    start = time.time()
    indices = SklearnNeighbors(pool).indices_deconv(pc, next_pts=pc_query, K=K)
    print(time.time() - start)
    print(indices)

    start = time.time()
    indices = open3d_knn_search(pc, pc_query, k=K)
    print(time.time() - start)
    print(indices)
    #
    # start = time.time()
    # indices = nearest_neighbors.knn_batch(pc, pc_query, K, omp=True)
    # print(time.time() - start)
    # print(indices)


if __name__ == '__main__':
    batch_size = 16
    num_points = 81920
    K = 16

    # class attribute, shared accross PtConv objects
    pool = mp.Pool(mp.cpu_count())
    # pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

    TEST_PATH = os.path.join('D:/develop/workstations/PCL_projects/cloud/trainedData')
    # extend = '.xyz'
    extend = '.ply'

    # file_list = DP.get_files_list(TEST_PATH, extend)
    # file = file_list[0]
    # pc = read_convert_to_array(file)
    # if extend == '.ply':
    #     pc = read_convert_to_array(file)
    # elif extend == '.xyz':
    #     pc = DP.load_pc_semantic3d(file, header=None, delim_whitespace=True)
    # else:
    #     exit(0)

    print(o3d.__version__)

    demo4()

    # for file in file_list:
    #     if extend == '.ply':
    #         pc = read_convert_to_array(file)
    #     elif extend == '.xyz':
    #         pc = DP.load_pc_semantic3d(file, header=None, delim_whitespace=True)
    #     else:
    #         continue
