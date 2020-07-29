from .base_detector import BaseDetector
from ..RandLA_Net.inference import inference


class RandLANetDetector(BaseDetector):
    def __init__(self):
        super(RandLANetDetector, self).__init__()

    def _config(self):
        pass

    def _read_pb_tensors(self):
        pass

    def _run_inference_local(self, data):
        return inference(data, need_sample=self.cfgs.NEED_SAMPLE,
                         use_votes=self.cfgs.USE_VOTES,
                         logger=self.logger)

    def _run_inference_server(self, data):
        error_message = 'server mode: {} has not been implemented yet...'.format(self.server_mode)
        BaseDetector.log_errors(message=error_message, error_type=NotImplementedError)
        return []
