import os
import sys
import traceback
from ..configs import cfgs
from ..help_utils.logger_utils import logger
from ..configs import finalize_configs
from ..help_utils.manager_utils.env_manager import collect_env_info

__all__ = ['register_detectors']


def register_detectors():
    try:
        update_dict = dict()
        if cfgs.DETECTOR not in ['RandLA', ]:
            logger.info("Environment Information:\n" + collect_env_info())
            text = "{} detector has not been support yet...".format(cfgs.DETECTOR)
            logger.error(text)
            raise NotImplementedError(text)

        if cfgs.DETECTOR == 'RandLA':
            from .randLA_detector import RandLANetDetector as Detector
            update_dict["USE_VOTES"] = False
            if cfgs.TOOLS_TYPE == "CLOUDVIEWER":
                update_dict["NEED_SAMPLE"] = True
            elif cfgs.TOOLS_TYPE == "OPEN3D":
                update_dict["NEED_SAMPLE"] = False
        else:
            logger.info("Environment Information:\n" + collect_env_info())
            text = 'model type : {} has not been implemented yet...'.format(cfgs.DETECTOR)
            logger.error(text)
            raise NotImplementedError(text)

        # update configuration parameters in cfg.py file
        cfgs.update_args(update_dict)
        # check the validation of configuration
        finalize_configs(logger)
        return Detector()
    except Exception as e:
        logger.exception(traceback.format_exc())
        sys.exit(0)
