import os
import time
import math
import random
import logging
import colorsys
import numpy as np
import pandas as pd
import cloudViewer as cv3d
import multiprocessing
from queue import Queue, Empty
from os.path import exists, isfile, join, splitext
from . import file_processing
from .helper_ply import read_ply
from .opencv import initialize_opencv


class Utility:
    @staticmethod
    def voxel_sampling(points, features=None, grid_size=0.1):
        cloud = cv3d.geometry.ccPointCloud()
        cloud.set_points(cv3d.utility.Vector3dVector(points))
        if features is not None:
            if np.max(features) > 1:
                cloud.set_colors(cv3d.utility.Vector3dVector(features / 255.))
            else:
                cloud.set_colors(cv3d.utility.Vector3dVector(features))
        grid_size = cloud.compute_resolution() * 2
        down_cloud = cloud.voxel_down_sample(grid_size)
        if features is not None:
            return np.asarray(down_cloud.get_points()), np.asarray(down_cloud.get_colors())
        else:
            return np.asarray(down_cloud.get_points())

    @staticmethod
    def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
        color = cv3d.io.read_image(color_file)
        depth = cv3d.io.read_image(depth_file)
        rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_trunc=config["max_depth"],
            convert_rgb_to_intensity=convert_rgb_to_intensity)
        return rgbd_image

    @staticmethod
    def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                               with_opencv, config):
        source_rgbd_image = Utility.read_rgbd_image(color_files[s], depth_files[s], True, config)
        target_rgbd_image = Utility.read_rgbd_image(color_files[t], depth_files[t], True, config)

        option = cv3d.odometry.OdometryOption()
        option.max_depth_diff = config["max_depth_diff"]
        if abs(s - t) != 1:
            # check opencv python package
            with_opencv = initialize_opencv()
            if with_opencv:
                from .opencv_pose_estimation import pose_estimation  # check opencv python package
                success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                        target_rgbd_image,
                                                        intrinsic, False)
                if success_5pt:
                    [success, trans, info] = cv3d.odometry.compute_rgbd_odometry(
                        source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                        cv3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                    return [success, trans, info]
            return [False, np.identity(4), np.identity(6)]
        else:
            odo_init = np.identity(4)
            [success, trans, info] = cv3d.odometry.compute_rgbd_odometry(
                source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                cv3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
            return [success, trans, info]

    @staticmethod
    def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                    depth_files, fragment_id, n_fragments,
                                    intrinsic, with_opencv, config):
        cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Error)
        pose_graph = cv3d.registration.PoseGraph()
        trans_odometry = np.identity(4)
        pose_graph.nodes.append(cv3d.registration.PoseGraphNode(trans_odometry))
        for s in range(sid, eid):
            for t in range(s + 1, eid):
                # odometry
                if t == s + 1:
                    print(
                        "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                        % (fragment_id, n_fragments - 1, s, t))
                    [success, trans,
                     info] = Utility.register_one_rgbd_pair(s, t, color_files, depth_files,
                                                                  intrinsic, with_opencv, config)
                    trans_odometry = np.dot(trans, trans_odometry)
                    trans_odometry_inv = np.linalg.inv(trans_odometry)
                    pose_graph.nodes.append(
                        cv3d.registration.PoseGraphNode(trans_odometry_inv))
                    pose_graph.edges.append(
                        cv3d.registration.PoseGraphEdge(s - sid,
                                                       t - sid,
                                                       trans,
                                                       info,
                                                       uncertain=False))

                # keyframe loop closure
                if s % config['n_keyframes_per_n_frame'] == 0 \
                        and t % config['n_keyframes_per_n_frame'] == 0:
                    print(
                        "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                        % (fragment_id, n_fragments - 1, s, t))
                    [success, trans,
                     info] = Utility.register_one_rgbd_pair(s, t, color_files, depth_files,
                                                                  intrinsic, with_opencv, config)
                    if success:
                        pose_graph.edges.append(
                            cv3d.registration.PoseGraphEdge(s - sid,
                                                           t - sid,
                                                           trans,
                                                           info,
                                                           uncertain=True))
        cv3d.io.write_pose_graph(
            join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
            pose_graph)

    @staticmethod
    def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                          n_fragments, pose_graph_name, intrinsic,
                                          config):
        pose_graph = cv3d.io.read_pose_graph(pose_graph_name)
        volume = cv3d.integration.ScalableTSDFVolume(
            voxel_length=config["tsdf_cubic_size"] / 512.0,
            sdf_trunc=0.04,
            color_type=cv3d.integration.TSDFVolumeColorType.RGB8)
        for i in range(len(pose_graph.nodes)):
            i_abs = fragment_id * config['n_frames_per_fragment'] + i
            print(
                "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
                (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
            rgbd = Utility.read_rgbd_image(color_files[i_abs], depth_files[i_abs], False, config)
            pose = pose_graph.nodes[i].pose
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


class IO:
    @staticmethod
    def read_image(file):
        img = cv3d.io.read_image(file)
        return img

    @staticmethod
    def read_mesh(file):
        mesh = cv3d.io.read_triangle_mesh(file)
        pc_xyz = np.asarray(mesh.get_vertices())
        pc_colors = np.asarray(mesh.get_vertex_colors())
        pc_triangles = np.asarray(mesh.get_triangles())
        if len(pc_colors) == 0:
            pc_colors.reshape(pc_xyz.shape)
        return np.hstack([pc_xyz, pc_triangles, pc_colors])

    @staticmethod
    def read_point_cloud(file):
        pcd = cv3d.io.read_point_cloud(file)
        pc_xyz = np.asarray(pcd.get_points())
        pc_colors = np.asarray(pcd.get_colors())
        if len(pc_colors) == 0:
            pc_colors.reshape(pc_xyz.shape)
        return np.hstack([pc_xyz, pc_colors])

    @staticmethod
    def read_convert_to_array(file):
        pc = read_ply(file)
        pc_xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T
        pc_colors = np.vstack((pc['red'], pc['green'], pc['blue'])).T
        return np.hstack([pc_xyz, pc_colors])

    @staticmethod
    def load_pc_semantic3d(filename, header=None, delim_whitespace=True, dtype=np.float32):
        pc_pd = pd.read_csv(filename, header=header, delim_whitespace=delim_whitespace, dtype=dtype)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)


class KNNNeighbors:
    thread_mode = 'thread'
    process_mode = 'process'
    async_mode = 'async'
    network_mode = 'machine'

    # https://github.com/ferventdesert/multi_yielder
    class Stop(Exception):
        "a flag when queue should stop"
        pass

    class Yielder(object):
        '''a yield context manager'''

        def __init__(self, dispose):
            self.dispose = dispose

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.dispose()

    @staticmethod
    def safe_queue_get(queue, is_stop_func=None, timeout=2):
        while True:
            if is_stop_func is not None and is_stop_func():
                return KNNNeighbors.Stop
            try:
                data = queue.get(timeout=timeout)
                return data
            except:
                continue

    @staticmethod
    def safe_queue_put(queue, item, is_stop_func=None, timeout=2):
        while True:
            if is_stop_func is not None and is_stop_func():
                return KNNNeighbors.Stop
            try:
                queue.put(item, timeout=timeout)
                return item
            except:
                continue

    @staticmethod
    def multi_yield(generator, customer_func, mode=thread_mode, worker_count=2, queue_size=10):
        """
        :param generator:  a iter object offer task seeds, can be array, generator
        :param customer_func:  your task func, get seed as parameter and yield result, if no result needed, yield None
        :param mode: three support mode: thread, gevent, process
        :param worker_count:
        :param queue_size:
        :return: a result generator
        """
        workers = []

        vebose = False

        def is_alive(process):
            if mode == KNNNeighbors.process_mode:
                return process.is_alive()
            elif mode == KNNNeighbors.thread_mode:
                return process.isAlive()
            return True

        class Stop_Wrapper():
            def __init__(self):
                self.stop_flag = False
                self.workers = []

            def is_stop(self):
                return self.stop_flag

            def stop(self):
                self.stop_flag = True
                for process in self.workers:
                    if isinstance(process, multiprocessing.Process):
                        process.terminate()

        stop_wrapper = Stop_Wrapper()

        def _boss(task_generator, task_queue, worker_count):
            for task in task_generator:
                item = KNNNeighbors.safe_queue_put(task_queue, task, stop_wrapper.is_stop)
                if item is KNNNeighbors.Stop:
                    if vebose:
                        print('downloader boss stop')
                    return
            for i in range(worker_count):
                task_queue.put(Empty)

            if vebose:
                print('worker boss finished')

        def _worker(task_queue, result_queue, gene_func):
            import time
            try:
                while not stop_wrapper.is_stop():
                    if task_queue.empty():
                        time.sleep(0.01)
                        continue
                    task = KNNNeighbors.safe_queue_get(task_queue, stop_wrapper.is_stop)
                    if not isinstance(task, np.ndarray):
                        if task == Empty:
                            result_queue.put(Empty)
                            break
                        if task == KNNNeighbors.Stop:
                            break
                    for item in gene_func(task):
                        item = KNNNeighbors.safe_queue_put(result_queue, item, stop_wrapper.is_stop)
                        if not isinstance(item, np.ndarray):
                            if item == KNNNeighbors.Stop:
                                break
                if vebose:
                    print('worker worker is stop')
            except Exception as e:
                logging.exception(e)
                if vebose:
                    print('worker exception, quit')

        def factory(func, args=None, name='task'):
            if args is None:
                args = ()
            if mode == KNNNeighbors.process_mode:
                return multiprocessing.Process(name=name, target=func, args=args)
            if mode == KNNNeighbors.thread_mode:
                import threading
                t = threading.Thread(name=name, target=func, args=args)
                t.daemon = True
                return t
            if mode == KNNNeighbors.async_mode:
                import gevent
                return gevent.spawn(func, *args)

        def queue_factory(size):
            if mode == KNNNeighbors.process_mode:
                return multiprocessing.Queue(size)
            elif mode == KNNNeighbors.thread_mode:
                return Queue(size)
            elif mode == KNNNeighbors.async_mode:
                from gevent import queue
                return queue.Queue(size)

        def should_stop():
            if not any([r for r in workers if is_alive(r)]):
                return True
            return stop_wrapper.is_stop()

        with KNNNeighbors.Yielder(stop_wrapper.stop):
            result_queue = queue_factory(queue_size)
            task_queue = queue_factory(queue_size)

            main = factory(_boss, args=(generator, task_queue, worker_count), name='_boss')
            for process_id in range(0, worker_count):
                name = 'worker_%s' % (process_id)
                p = factory(_worker, args=(task_queue, result_queue, customer_func), name=name)
                workers.append(p)
            main.start()
            stop_wrapper.workers = workers[:]
            stop_wrapper.workers.append(main)
            for r in workers:
                r.start()
            count = 0
            while not should_stop():
                data = KNNNeighbors.safe_queue_get(result_queue, should_stop)
                if data is Empty:
                    count += 1
                    if count == worker_count:
                        break
                    continue
                if data is KNNNeighbors.Stop:
                    break
                else:
                    yield data

    @staticmethod
    def multiply_knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        assert len(support_pts.shape) == 3
        assert len(query_pts.shape) == 3

        batch_number = query_pts.shape[0]
        batch_points = query_pts.shape[1]
        neighbor_idx = np.zeros((batch_number, batch_points, k))

        def task(data):
            batch_data = data[0]
            batch_query = data[1]
            indices = np.zeros((batch_points, k))
            kd_tree = cv3d.geometry.KDTreeFlann(np.swapaxes(batch_data, 0, 1))
            for j, pts in enumerate(batch_query):
                [dummy, idx, _] = kd_tree.search_knn_vector_3d(pts, k)
                indices[j, :] = np.asarray(idx)
            yield indices

        i = 0
        dat = zip(support_pts, query_pts)
        for item in KNNNeighbors.multi_yield(dat, task, KNNNeighbors.thread_mode, multiprocessing.cpu_count()):
            neighbor_idx[i, :, :] = item
            i += 1
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def single_knn_search(support_pts, query_pts, k):
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
        neighbor_idx = np.zeros((batch_number, batch_size, k))
        for iter in range(batch_number):
            kd_tree = cv3d.geometry.KDTreeFlann(np.swapaxes(support_pts[iter], 0, 1))
            for i, pts in enumerate(query_pts[iter]):
                [dummy, idx, _] = kd_tree.search_knn_vector_3d(pts, k)
                neighbor_idx[iter, i, :] = np.asarray(idx)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def test():
        def xprint(x):
            """
            mock a long time task
            """
            time.sleep(1)
            yield x * x

        i = 0
        for item in KNNNeighbors.multi_yield(range(100), xprint, KNNNeighbors.process_mode, 3):
            print(item)
            i += 1
            if i > 10:
                break

    @staticmethod
    def test2():
        TEST_PATH = os.path.join('/media/yons/data/dataset/pointCloud/data/ownTrainedData/test')
        # extend = '.xyz'
        extend = '.ply'
        THREAD_NUMBER = multiprocessing.cpu_count()
        num_points = 81920
        K = 16

        fileList = file_processing.get_files_list(TEST_PATH, extend)
        for file in fileList:
            if extend == '.ply':
                pc = IO.read_convert_to_array(file)
            elif extend == '.xyz':
                pc = IO.load_pc_semantic3d(file, header=None, delim_whitespace=True)
            else:
                continue

        pc = pc[:, :3].astype(np.float32)
        batch_size = math.floor(pc.shape[0] / num_points)
        pc = pc[:batch_size * num_points]
        pc = pc.reshape([batch_size, num_points, 3])
        pc_query = pc[:, :40000, :]

        # nearest neighbours
        start = time.time()
        indexs = KNNNeighbors.multiply_knn_search(pc, pc_query, k=K)
        print(time.time() - start)
        print(indexs)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = cv3d.geometry.ccPointCloud()
        pc.set_points(cv3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3]))
        if pc_xyzrgb.shape[1] == 3:
            cv3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  # 0-255
            pc.set_colors(cv3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.))
        else:
            pc.set_colors(cv3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6]))
        cv3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, fix_color_num=None):
        if fix_color_num is not None:
            ins_colors = Plot.random_colors(fix_color_num + 1, seed=2)
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)  # cls 14

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if fix_color_num is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            # bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0])
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1])
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2])
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins


if __name__ == '__main__':
    KNNNeighbors.test2()
