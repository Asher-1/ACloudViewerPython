# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/22 17:03
# @Author  : ludahai
# @FileName: sklearn_utils.py
# @Software: PyCharm


import numpy as np
from sklearn.neighbors import BallTree


def mp_indices_conv(pts, K=16):
    tree = BallTree(pts, leaf_size=2)
    _, indices = tree.query(pts, k=K)
    return indices


def mp_indices_deconv(pts, pts_next, K):
    tree = BallTree(pts, leaf_size=2)
    _, indices = tree.query(pts_next, k=K)
    return indices


def mp_indices_conv_reduction(pts, K, npts):
    tree = BallTree(pts, leaf_size=2)
    used = np.zeros(pts.shape[0])
    current_id = 0
    indices = []
    pts_n = []
    for ptid in range(npts):

        # index = np.random.randint(pts.shape[0])
        possible_ids = np.argwhere(used == current_id).ravel().tolist()
        while (len(possible_ids) == 0):
            current_id = used.min()
            possible_ids = np.argwhere(used == current_id).ravel().tolist()

        index = possible_ids[np.random.randint(len(possible_ids))]

        # pick a point
        pt = pts[index]

        # perform the search
        dist, ids = tree.query([pt], k=K)
        ids = ids[0]

        used[ids] += 1
        used[index] += 1e7

        indices.append(ids.tolist())
        pts_n.append(pt)

    pts_n = np.array(pts_n)

    return indices, pts_n


class SklearnNeighbors(object):

    def __init__(self, pool):
        super(object, self).__init__()
        self.pool = pool

    def indices_conv_reduction(self, input_pts, K, npts):
        pts = [(input_pts[i], K, npts[i]) for i in range(input_pts.shape[0])]
        indices = self.pool.starmap(mp_indices_conv_reduction, pts)
        indices, pts = zip(*indices)
        return indices

    def indices_conv(self, input_pts, K):
        pts = [(input_pts[i], K) for i in range(input_pts.size(0))]
        indices = self.pool.starmap(mp_indices_conv, pts)
        return indices

    def indices_deconv(self, pts, next_pts, K):
        pts_ = [(pts[i], next_pts[i], K) for i in range(pts.shape[0])]
        indices = self.pool.starmap(mp_indices_deconv, pts_)
        return np.array(indices).astype(np.int32)
