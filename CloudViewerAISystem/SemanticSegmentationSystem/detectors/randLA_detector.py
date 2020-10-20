import os
import math
from .base_detector import BaseDetector
from ..RandLA_Net.inference import inference


class RandLANetDetector(BaseDetector):
    def __init__(self):
        super(RandLANetDetector, self).__init__()

    @staticmethod
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

    def _config(self):
        gpu_index, updated_infer_batch_size = RandLANetDetector.config_device(self.logger)
        self.gpu_index = gpu_index
        self.infer_batch_size = updated_infer_batch_size
        self.logger.info("using GPU index {}".format(gpu_index))

    def _read_pb_tensors(self):
        pass

    def _run_inference_local(self, data):
        return inference(data, need_sample=self.cfgs.NEED_SAMPLE,
                         use_votes=self.cfgs.USE_VOTES,
                         logger=self.logger,
                         test_batch_size=self.infer_batch_size,
                         gpu_index=self.gpu_index)

    def _run_inference_server(self, data):
        error_message = 'server mode: {} has not been implemented yet...'.format(self.server_mode)
        BaseDetector.log_errors(message=error_message, error_type=NotImplementedError)
        return []
