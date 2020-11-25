# -*- coding: utf-8 -*-
import os
import six
import json
from tabulate import tabulate
from .label_name_dict import label_dict

__all__ = ['config', 'finalize_configs']


def _log_errors(logger, error_message, error_type):
    logger.error(error_message)
    raise error_type(error_message)


def get_gpu_info():
    import pynvml
    pynvml.nvmlInit()
    gpu_num = pynvml.nvmlDeviceGetCount()
    res = {}
    for i in range(gpu_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        index = "GPU {} INFO".format(i)
        info = {}
        info["total"] = "{} MB".format(str(meminfo.total / 1024 ** 2))
        info["used"] = "{} MB".format(str(meminfo.used / 1024 ** 2))
        info["free"] = "{} MB".format(str(meminfo.free / 1024 ** 2))
        res[index] = info
    return json.dumps(res, ensure_ascii=False, sort_keys=False, indent=4)


class AttrDict(object):
    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        if name.startswith('_'):
            # Do not mess with internals. Otherwise copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        # return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)
        data = []
        for k, v in self.to_dict().items():
            data.append((str(k), str(v)))
        return tabulate(data)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """Update from command line args. """
        if isinstance(args, list) or isinstance(args, tuple):
            for cfg in args:
                keys, v = cfg.split('=', maxsplit=1)
                keylist = keys.split('.')

                dic = self
                for i, k in enumerate(keylist[:-1]):
                    assert k in dir(dic), "Unknown config key: {}".format(keys)
                    dic = getattr(dic, k)
                key = keylist[-1]

                oldv = getattr(dic, key)
                if not isinstance(oldv, str):
                    v = eval(v)
                setattr(dic, key, v)
        elif isinstance(args, dict):
            for keys, v in args.items():
                keylist = keys.split('.')
                dic = self
                for i, k in enumerate(keylist[:-1]):
                    assert k in dir(dic), "Unknown config key: {}".format(keys)
                    dic = getattr(dic, k)
                key = keylist[-1]
                setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()

_C = config  # short alias to avoid coding

# path configuration --------------------------------------
_C.ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
_C.TEST_PATH = os.path.join(_C.ROOT_PATH, 'test', 'test_clouds')
_C.TEST_RESULT = os.path.join(_C.ROOT_PATH, 'test', 'test_result')
_C.LOG_PATH = os.path.join(_C.ROOT_PATH, 'log')
_C.LOG_FILE = os.path.join(_C.LOG_PATH, 'log.log')

# system configuration ---------------------------------------------------
_C.DETECTOR = 'RandLA'  # 'RandLA',
_C.RUNNING_MODE = 'LOCAL'  # 'SERVER' or 'LOCAL'
_C.SERVER_MODE = 'GRPC'  # 'GRPC' or 'REST'
_C.GRPC_SERVER_URL = 'localhost:9000'
_C.REST_SERVER_URL = 'http://localhost:9001/v1/models/HelmetDetector:predict'
_C.SNIPER_SERVER_URL = "http://localhost:9002/helmetdetector"

# detection configuration ------------------------------------------------
gpu_info = get_gpu_info()
_C.GPU_INFO = gpu_info
_C.GPU_MODE = 0
_C.NUM_CLASSES = 20
_C.MIN_POINTS = 100
_C.VOXEL_SIZE = 0.03  # 0.02 default
_C.NAME_LABEL_MAP = label_dict.NAME_LABEL_MAP
_C.LABEL_NAME_MAP = label_dict.LABEL_NAME_MAP
_C.GPU_MEMORY_FRACTION = 0.5
_C.NMS_THRESHOLD = 0.5
_C.VISUALIZATION = True
_C.NEED_SAMPLE = True
_C.USE_VOTES = False
_C.TOOLS_TYPE = "CLOUDVIEWER"  # "OPEN3D"


def finalize_configs(logger):
    """
    Run some sanity checks, and populate some configs from others
    """
    _C.freeze(False)  # populate new keys now
    if isinstance(_C.DATA.VAL, six.string_types):  # support single string (the typical case) as well
        _C.DATA.VAL = (_C.DATA.VAL,)

    if not (isinstance(_C.NUM_CLASSES, int) and _C.NUM_CLASSES >= 1):
        error_message = "invalid class number, please check NUM_CLASSES..."
        _log_errors(logger, error_message=error_message, error_type=ValueError)

    _C.freeze()
    logger.info('Semantic Segmentation System Configuration: \n' + str(_C))
