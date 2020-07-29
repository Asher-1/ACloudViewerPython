import os
import time
import math
import numpy as np
import open3d as o3d
from helper_ply import read_ply, write_ply
from helper_tool import DataProcessing as DP
from manager_utils.sklearn_neighbors import SklearnNeighbors
import multiprocessing as mp


def read_convert_to_array(file):
    pc = read_ply(file)
    pc_xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T
    pc_colors = np.vstack((pc['red'], pc['green'], pc['blue'])).T
    return np.hstack([pc_xyz, pc_colors])


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
    print(file)
    if extend == '.ply':
        pc = read_convert_to_array(file)
    elif extend == '.xyz':
        pc = DP.load_pc_semantic3d(file, header=None, delim_whitespace=True)
    else:
        exit(0)

    print("pc shape {}".format(pc.shape))

    sub_points, sub_colors = DP.open3d_voxel_sampling(pc[:, :3].astype(np.float32),
                                                      pc[:, 3:6].astype(np.uint8),
                                                      grid_size=0.02)
    print("down sampling shape {}".format(sub_points.shape))
    print(sub_points)

    # sub_ply_file = os.path.join(os.path.dirname(file), "sub_sampling.ply")
    # write_ply(sub_ply_file, [sub_points, sub_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])


if __name__ == '__main__':
    batch_size = 16
    num_points = 81920
    K = 16

    # class attribute, shared accross PtConv objects
    pool = mp.Pool(4)
    # pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

    TEST_PATH = os.path.join('/media/yons/data/dataset/pointCloud/data/ownTrainedData/test')
    extend = '.xyz'
    # extend = '.ply'
    print("open3D vision: " + o3d.__version__)
    demo4()
