#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys


def print_exception_traceback_detail():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # traceback_details = {
    #     'filename': exc_traceback.tb_frame.f_code.co_filename,
    #     'lineno': exc_traceback.tb_lineno,
    #     'name': exc_traceback.tb_frame.f_code.co_name,
    #     'type': exc_type.__name__,
    #     'message': exc_value.message, # or see traceback._some_str()
    # }
    print("exception type: {0}".format(exc_type.__name__))
    print("exception message: {0}".format(exc_value))
    print("exception line number: {0}".format(exc_traceback.tb_lineno))
    print("exception filename: {0}".format(exc_traceback.tb_frame.f_code.co_filename))
    print("exception name: {0}".format(exc_traceback.tb_frame.f_code.co_name))
