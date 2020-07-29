#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import traceback
import numpy as np
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from ..configs import cfgs
if cfgs.TOOLS_TYPE == "CLOUDVIEWER":
    from SemanticSegmentationSystem.help_utils import cloudviewer_utils as tools
elif cfgs.TOOLS_TYPE == "OPEN3D":
    from SemanticSegmentationSystem.help_utils import open3d_utils as tools
from ..help_utils.logger_utils import logger


class BaseDetector(metaclass=ABCMeta):
    def __init__(self):
        self._config_for_all()
        if self.mode == 'LOCAL':
            self._load_model()
        elif self.mode == 'SERVER':
            message = 'server mode: {} has not been implemented yet...'.format(self.server_mode)
            BaseDetector.log_errors(message=message, error_type=NotImplementedError)
        else:
            message = 'running mode: {} has not been implemented yet...'.format(self.mode)
            BaseDetector.log_errors(message=message, error_type=NotImplementedError)

        self._config()

    def _config_for_all(self):
        self.cfgs = cfgs
        self.logger = logger
        self.mode = cfgs.RUNNING_MODE
        self.gpu_mode = cfgs.GPU_MODE
        # self.pool = mp.Pool(mp.cpu_count())

        # gpu configuration
        if self.gpu_mode == 0:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif self.gpu_mode == 1:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.detector = cfgs.DETECTOR
        self.server_mode = cfgs.SERVER_MODE
        self.num_classes = cfgs.NUM_CLASSES
        self.visualization = cfgs.VISUALIZATION
        self.nms_threshold = cfgs.NMS_THRESHOLD
        self.gpu_memory_fraction = cfgs.GPU_MEMORY_FRACTION

    def _config_for_rest(self):
        self.server_url = cfgs.REST_SERVER_URL

    @abstractmethod
    def _config(self):
        """
        for local or server detection
        Override this method to define a specific config behavior, to be used with detectors.
        Note that derived class (eg. CascadeDetector) must override this method otherwise type errors will occur
        :return: None
        """
        pass

    @abstractmethod
    def _read_pb_tensors(self):
        """
        for local detection
        Override this method to define a specific read_pb_tensors behavior, to be used with detectors.
        Note that derived class (eg. CascadeDetector) must override this method otherwise type errors will occur
        :return: None
        """
        pass

    @abstractmethod
    def _run_inference_local(self, data):
        """
        for local detection
        Override this method to define a specific run_inference_local behavior, to be used with detectors.
        Note that derived class (eg. CascadeDetector) must override this method otherwise type errors will occur
        :param image_np: the image array (RGB) that to be detected
        :return: the detection results numpy array
        """
        pass

    @abstractmethod
    def _run_inference_server(self, data):
        """
        for server detection
        Override this method to define a specific run_inference_server behavior, to be used with detectors.
        Note that derived class (eg. CascadeDetector) must override this method otherwise type errors will occur.
        :param response: the server api [GRPC or REST] response values
        :param image_np: the image array (RGB) that to be detected
        :return: the detection results numpy array
        """
        pass

    def _load_model(self):
        pass

    def run(self, data):
        """
        the main input api, used for get batch data and return detection result
        :param data: inference the format of return type
        :return: detection result
        """
        start = time.time()
        pc_list = []
        if data is None or len(data) < 1:
            message = 'data is None or len(data) < 1: {}'.format(data)
            BaseDetector.log_errors(message=message, level='warning')
            return {'result': [], 'timeTake': 0, 'state': message}
        elif isinstance(data[0], list):
            logger.info("input list data shape {}".format(len(data)))
            pc_list = [np.asarray(d) for d in data]
        elif isinstance(data[0], np.ndarray):
            logger.info("input array data shape {}".format(data[0].shape))
            pc_list = data
        elif isinstance(data[0], str):
            for file in data:
                pc = tools.IO.read_point_cloud(file)
                if pc.is_empty():
                    logger.info("read point cloud : {} failed!".format(file))
                    continue
                np.hstack([np.asarray(pc.points), np.asarray(pc.colors)])
                pc_list.append(pc)
        else:
            message = 'invalid format of data : {}'.format(data)
            BaseDetector.log_errors(message=message, level='warning')
            return {'result': [], 'timeTake': 0, 'state': message}

        detect_result = []
        try:
            detect_result = self._sementic_segmentation(pc_list)
        except Exception as e:
            message = str(traceback.format_exc())
            BaseDetector.log_errors(message=message, level='warning')
            return {'result': [], 'timeTake': 0, 'state': message}

        time_take = time.time() - start
        return {'result': detect_result, 'timeTake': round(time_take, 4), 'state': 'success'}

    def _sementic_segmentation(self, data):
        """
        preprocess image data according data type and
        call inference api according running mode [local, server]
        :param data: the point clouds data
        :return: list
        """
        res = []
        if self.mode == 'LOCAL':
            res = self._inference_on_local(data)
        elif self.mode == 'SERVER':
            res = self._inference_on_server(data)
        else:
            message = 'typeError: the running mode must be [server or local], but got {}'.format(self.mode)
            BaseDetector.log_errors(message=message)
            raise NotImplementedError(message)
        return res

    def _inference_on_local(self, data):
        return self._run_inference_local(data)

    def _inference_on_server(self, data):
        # sending post request to TensorFlow Serving server
        return self._run_inference_server(data)

    @staticmethod
    def log_errors(message, error_type=None, level='error'):
        message += "[{}]".format(int(sys._getframe().f_back.f_lineno))
        if level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'info':
            logger.info(message)
        elif level == 'debug':
            logger.debug(message)
        elif level == 'exception':
            logger.exception(message)
        else:
            logger.error("level: {} has not been implemented yet...".format(level))
            raise NotImplementedError("level: {} has not been implemented yet...".format(level))

        if error_type is not None:
            raise error_type(message)
