# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/6/4 9:15
# @Author  : ludahai
# @FileName: timer_wraper.py
# @Software: PyCharm

import time
import timeit
import datetime
from functools import wraps

logfile_path = 'D:/develop/workstations/resource/datasets/helmet_db/data/test/head_out/log.txt'


def timer_wrapper(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = "{}\t{}-->running time: {} seconds".format(nowTime, func.__name__, str(t1 - t0))
        print(text)
        return result

    return function_timer


def clock_wrapper(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        t0 = time.clock()
        result = func(*args, **kwargs)
        t1 = time.clock()
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = "{}\t{}-->running time: {} seconds".format(nowTime, func.__name__, str(t1 - t0))
        print(text)
        return result

    return function_timer


def cross_timer_wrapper(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        t1 = timeit.default_timer()
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = "{}\t{}-->running time: {} seconds".format(nowTime, func.__name__, str(t1 - t0))
        print(text)
        return result

    return function_timer
