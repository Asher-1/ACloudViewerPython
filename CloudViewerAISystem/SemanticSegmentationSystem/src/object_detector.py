#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
from ..help_utils.logger_utils import logger
from ..detectors.register_detectors import register_detectors
from ..help_utils.manager_utils.env_manager import collect_env_info


class ObjectDetector(object):
    def __init__(self):
        self.logger = logger
        self.detector = register_detectors()
        if self.detector is None:
            logger.info("Environment Information:\n" + collect_env_info())
            text = "the detector obtained from register_detectors is None"
            logger.warning(text)
            raise ValueError(text)

    def semantic_segmentation(self, *data):
        try:
            res = self.detector.run(*data)
            return res
        except Exception as e:
            logger.info("Environment Information:\n" + collect_env_info())
            logger.exception(e)
            sys.exit(0)
