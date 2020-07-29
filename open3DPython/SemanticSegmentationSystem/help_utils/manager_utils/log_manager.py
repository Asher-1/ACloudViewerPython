#!/usr/bin/python
# -*- coding: UTF-8 -*-


'''
we use this script to generate log we want
'''

import sys
import logging
from termcolor import colored
from logging.handlers import TimedRotatingFileHandler

__all__ = ['Logger']


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRNNING ' + msg, 'yellow', attrs=['blink'])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERROR ' + msg, 'red', attrs=['blink', 'underline'])
        elif record.levelno == logging.DEBUG or record.levelno == logging.INFO:
            fmt = date + ' ' + colored('INFO ' + msg, 'blue', attrs=['bold'])
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


class Logger(object):
    def __init__(self, log_name):
        # 文件的命名
        self.logname = log_name
        self.logger = logging.getLogger('AI')
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        # 创建一个FileHandler，用于写到本地
        self.fh = TimedRotatingFileHandler(self.logname, when="D", encoding='utf-8', interval=1, backupCount=7)
        self.fh.suffix = "%Y-%m-%d_%H-%M-%S.log"
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(logging.Formatter('[%(asctime)s @%(filename)s:%(lineno)d] %(levelname)s: %(message)s'))
        self.logger.addHandler(self.fh)

        # 创建一个StreamHandler,用于输出到控制台
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.DEBUG)
        self.ch.setFormatter(_MyFormatter(datefmt='%y-%m-%d %H:%M:%S'))
        self.logger.addHandler(self.ch)

    # def __console(self, level, message):
    #
    #     if level == 'info':
    #         self.logger.info(message)
    #     elif level == 'debug':
    #         self.logger.debug(message)
    #     elif level == 'warning':
    #         self.logger.warning(message)
    #     elif level == 'error':
    #         self.logger.error(message)
    #     elif level == 'critical':
    #         self.logger.critical(message)
    #     elif level == 'exception':
    #         self.logger.exception(message)
    #     elif level == 'setLevel':
    #         self.logger.setLevel(message)
    #     elif level == 'addFilter':
    #         self.logger.addFilter(message)
    #
    #     # 这两行代码是为了避免日志输出重复问题
    #     self.logger.removeHandler(ch)
    #     self.logger.removeHandler(fh)
    #     # 关闭打开的文件
    #     fh.close()

    # def debug(self, message):
    #     self.__console('debug', message)
    #
    # def info(self, message):
    #     self.__console('info', message)
    #
    # def warning(self, message):
    #     self.__console('warning', message)
    #
    # def error(self, message):
    #     self.__console('error', message)
    #
    # def critical(self, message):
    #     self.__console('critical', message)
    #
    # def exception(self, message):
    #     self.__console('exception', message)
    #
    # def set_level(self, message):
    #     self.__console('setLevel', message)
    #
    # def add_filter(self, message):
    #     self.__console('addFilter', message)
