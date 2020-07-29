#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import six
import re
import psutil
import logging
import numpy as np
from tabulate import tabulate
from collections import defaultdict
from .nvml_manager import NVMLContext
from contextlib import contextmanager

logging.getLogger().setLevel(logging.INFO)

if six.PY2:
    import subprocess32 as subprocess
else:
    import subprocess


def subproc_call(cmd, timeout=None):
    """
    Execute a command with timeout, and return STDOUT and STDERR

    Args:
        cmd(str): the command to execute.
        timeout(float): timeout in seconds.

    Returns:
        output(bytes), retcode(int). If timeout, retcode is -1.
    """
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT,
            shell=True, timeout=timeout)
        return output, 0
    except subprocess.TimeoutExpired as e:
        logging.warn("Command '{}' timeout!".format(cmd))
        logging.warn(e.output.decode('utf-8'))
        return e.output, -1
    except subprocess.CalledProcessError as e:
        logging.warn("Command '{}' failed, return code={}".format(cmd, e.returncode))
        logging.warn(e.output.decode('utf-8'))
        return e.output, e.returncode
    except Exception:
        logging.warn("Command '{}' failed to run.".format(cmd))
        return "", -2


@contextmanager
def change_env(name, val):
    """
    Args:
        name(str), val(str):

    Returns:
        a context where the environment variable ``name`` being set to
        ``val``. It will be set back after the context exits.
    """
    oldval = os.environ.get(name, None)
    os.environ[name] = val
    yield
    if oldval is None:
        del os.environ[name]
    else:
        os.environ[name] = oldval


def find_library(name):
    """
    Similar to `from ctypes.util import find_library`, but try
    to return full path if possible.
    """
    from ctypes.util import find_library

    if os.name == "posix" and sys.platform == "darwin":
        # on Mac, ctypes already returns full path
        return find_library(name)

    def _use_proc_maps(name):
        """
        Find so from /proc/pid/maps
        Only works with libraries that has already been loaded.
        But this is the most accurate method -- it finds the exact library that's being used.
        """
        procmap = os.path.join('/proc', str(os.getpid()), 'maps')
        if not os.path.isfile(procmap):
            return None
        with open(procmap, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                sofile = line[-1]

                basename = os.path.basename(sofile)
                if 'lib' + name + '.so' in basename:
                    if os.path.isfile(sofile):
                        return os.path.realpath(sofile)

    # The following two methods come from https://github.com/python/cpython/blob/master/Lib/ctypes/util.py
    def _use_ld(name):
        """
        Find so with `ld -lname -Lpath`.
        It will search for files in LD_LIBRARY_PATH, but not in ldconfig.
        """
        cmd = "ld -t -l{} -o {}".format(name, os.devnull)
        ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
        for d in ld_lib_path.split(':'):
            cmd = cmd + " -L " + d
        result, ret = subproc_call(cmd + '|| true')
        expr = r'[^\(\)\s]*lib%s\.[^\(\)\s]*' % re.escape(name)
        res = re.search(expr, result.decode('utf-8'))
        if res:
            res = res.group(0)
            if not os.path.isfile(res):
                return None
            return os.path.realpath(res)

    def _use_ldconfig(name):
        """
        Find so in `ldconfig -p`.
        It does not handle LD_LIBRARY_PATH.
        """
        with change_env('LC_ALL', 'C'), change_env('LANG', 'C'):
            ldconfig, ret = subproc_call("ldconfig -p")
            ldconfig = ldconfig.decode('utf-8')
            if ret != 0:
                return None
        expr = r'\s+(lib%s\.[^\s]+)\s+\(.*=>\s+(.*)' % (re.escape(name))
        res = re.search(expr, ldconfig)
        if not res:
            return None
        else:
            ret = res.group(2)
            return os.path.realpath(ret)

    if sys.platform.startswith('linux'):
        return _use_proc_maps(name) or _use_ld(name) or _use_ldconfig(name) or find_library(name)

    return find_library(name)  # don't know what to do


def get_num_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """

    def warn_return(ret, message):
        try:
            import tensorflow as tf
        except ImportError:
            return ret

        built_with_cuda = tf.test.is_built_with_cuda()
        if not built_with_cuda and ret > 0:
            logging.warn(message + "But TensorFlow was not built with CUDA support and could not use GPUs!")
        return ret

    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env:
        return warn_return(len(env.split(',')), "Found non-empty CUDA_VISIBLE_DEVICES. ")
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code == 0:
        output = output.decode('utf-8')
        return warn_return(len(output.strip().split('\n')), "Found nvidia-smi. ")
    try:
        # Use NVML to query device properties
        with NVMLContext() as ctx:
            return warn_return(ctx.num_devices(), "NVML found nvidia devices. ")
    except Exception:
        # Fallback
        logging.info("Loading local devices by TensorFlow ...")

        try:
            import tensorflow as tf
            # available since TF 1.14
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        except AttributeError:
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            # Note this will initialize all GPUs and therefore has side effect
            # https://github.com/tensorflow/tensorflow/issues/8136
            gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return len(gpu_devices)


def collect_env_info():
    """
    Returns:
        str - a table contains important information about the environment
    """
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("Numpy", np.__version__))
    has_cuda = False
    try:
        import tensorflow as tf
        tfv1 = tf.compat.v1
        data.append(("TensorFlow", tfv1.VERSION + "/" + tfv1.GIT_VERSION))
        data.append(("TF Compiler Version", tfv1.COMPILER_VERSION))
        has_cuda = tf.test.is_built_with_cuda()
        data.append(("TF CUDA support", has_cuda))
    except Exception as e:
        logging.warning(str(e) + '--->\t cannot find tensorflow package...')

    try:
        from tensorflow.python.framework import test_util
        data.append(("TF MKL support", test_util.IsMklEnabled()))
    except Exception:
        pass

    try:
        from tensorflow.python.framework import test_util
        data.append(("TF XLA support", test_util.is_xla_enabled()))
    except Exception:
        pass

    if has_cuda:
        data.append(("Nvidia Driver", find_library("nvidia-ml")))
        data.append(("CUDA", find_library("cudart")))
        data.append(("CUDNN", find_library("cudnn")))
        data.append(("NCCL", find_library("nccl")))

        # List devices with NVML
        data.append(("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", str(None))))
        try:
            devs = defaultdict(list)
            with NVMLContext() as ctx:
                for idx, dev in enumerate(ctx.devices()):
                    devs[dev.name()].append(str(idx))

            for devname, devids in devs.items():
                data.append(
                    ("GPU " + ",".join(devids), devname))
        except Exception:
            data.append(("GPU", "Not found with NVML"))

    vram = psutil.virtual_memory()
    data.append(("Free RAM", "{:.2f}/{:.2f} GB".format(vram.available / 1024 ** 3, vram.total / 1024 ** 3)))
    data.append(("CPU Count", psutil.cpu_count()))

    # Other important dependencies:
    try:
        import cv2
        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass

    import msgpack
    data.append(("msgpack", ".".join([str(x) for x in msgpack.version])))

    has_prctl = True
    try:
        import prctl
        _ = prctl.set_pdeathsig  # noqa
    except Exception:
        has_prctl = False
    data.append(("python-prctl", has_prctl))

    return tabulate(data)
