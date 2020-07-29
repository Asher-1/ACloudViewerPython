#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  insightface
FILE_NAME    :  logger_utils
AUTHOR       :  DAHAI LU
TIME         :  2019/7/29 下午7:28
PRODUCT_NAME :  PyCharm
"""
from ..configs import cfgs
from .manager_utils.log_manager import Logger

__all__ = ['logger']
logger = Logger(cfgs.LOG_FILE).logger
