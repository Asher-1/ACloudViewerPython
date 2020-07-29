#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  RandLA-Net
FILE_NAME    :  inference
AUTHOR       :  DAHAI LU
TIME         :  2020/5/12 下午2:42
PRODUCT_NAME :  PyCharm
================================================================
"""

import os
import sys
import time
import math
import numpy as np

from .DataSet import DataSetSemantic3D
from .helper_tool import ConfigSemantic3D
from .helper_tool import DataProcessing as DP
from .RandLANet_Inference import Network as NetworkInference
from .Semantic3D_Inference import ModelTester as ModelInference


def inference(data=None, need_sample=True, use_votes=False, pool=None, logger=None):
    logger.info("need_sample: {}; use_votes: {};".format(
        "True" if need_sample else "false",
        "True" if use_votes else "false"))

    DP.pool = pool

    config = ConfigSemantic3D()
    gpu_index, updated_infer_batch_size = config_device(logger)
    logger.info("using GPU index {}".format(gpu_index))
    config.test_batch_size = updated_infer_batch_size
    logger.info("updated test batch size {}".format(config.test_batch_size))

    ROOT_PATH = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(ROOT_PATH, 'snapshots/snap-29501')
    chosen_snap = MODEL_PATH

    #  step 2. tf data preparation
    t1 = time.time()
    dataset = DataSetSemantic3D(need_sample=need_sample, leaf_size=50, config=config, logger=logger)
    dataset.set_input_clouds(data)
    dataset.init_input_pipeline()
    t2 = time.time()
    logger.info('[tf data preparation] Done in {:.1f} s\n'.format(t2 - t1))

    #  step 3. construct network
    t1 = time.time()
    model = NetworkInference(dataset)
    t2 = time.time()
    logger.info('[construct network] Done in {:.1f} s\n'.format(t2 - t1))

    #  step 4. init network parameters
    t1 = time.time()
    tester = ModelInference(logger=logger, restore_snap=chosen_snap, on_cpu=False if gpu_index != -1 else True)
    t2 = time.time()
    logger.info('[init network parameters] Done in {:.1f} s\n'.format(t2 - t1))

    #  step 5. inference
    preds_list = tester.inference(model, dataset, use_votes=use_votes, num_votes=50)

    logger.info("pred shape : {}".format(np.array(preds_list).shape))
    return preds_list


def config_device(logger):
    gpu_index = -1
    batch_size = 2
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_num = pynvml.nvmlDeviceGetCount()
        logger.info("GPU Total Count： " + str(gpu_num))  # 显示有几块GPU
        free_memory_list = []
        for i in range(gpu_num):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory_list.append(meminfo.free / 1024 ** 2)

        max_gpu_memory = max(free_memory_list)
        gpu_index = free_memory_list.index(max_gpu_memory)
        batch_size = math.ceil(max_gpu_memory / 1024)
    except Exception as e:
        gpu_index = -1
        logger.warning(str(e) + '--->\t cannot get gpu info...')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    return gpu_index, batch_size


if __name__ == '__main__':
    TEST_PATH = os.path.join('/media/yons/data/dataset/pointCloud/data/ownTrainedData/test')