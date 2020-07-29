from helper_tool import Plot
import numpy as np
import os
from os.path import join, exists
from helper_ply import read_ply
from helper_tool import DataProcessing as DP

# DATA_PATH = '/media/yons/data/dataset/pointCloud/data/semantic3d'
# PRE_PATH = '/media/yons/data/dataset/pointCloud/data/semantic3d/test/predictions'

DATA_PATH = '/media/yons/data/dataset/pointCloud/data/ownTrainedData'
PRE_PATH = DATA_PATH + '/predictions'

if __name__ == '__main__':
    ##################
    # Visualize data #
    ##################
    sub_pc_folder = join(DATA_PATH, 'original_ply')
    cloud_names = [file_name[:-14] for file_name in os.listdir(PRE_PATH) if file_name[-7:] == '.labels']

    scene_names = []
    label_names = []
    for pc_name in cloud_names:
        if exists(join(sub_pc_folder, pc_name + '.ply')):
            scene_names.append(join(sub_pc_folder, pc_name + '.ply'))
            label_names.append(join(PRE_PATH, pc_name + '_result.labels'))

    for i in range(len(scene_names)):
        print('scene:', scene_names[i])
        data = read_ply(scene_names[i])
        data = np.vstack((data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'])).T
        pc = data[:, :6].astype(np.float32)
        print('scene point number', pc.shape)
        sem_pred = DP.load_label_semantic3d(label_names[i])
        sem_pred.astype(np.float32)

        ## plot
        Plot.draw_pc(pc_xyzrgb=pc[:, 0:6])
        sem_ins_labels = np.unique(sem_pred)
        print('sem_ins_labels: ', sem_ins_labels)
        Plot.draw_pc_sem_ins(pc_xyz=pc[:, 0:3], pc_sem_ins=sem_pred)
