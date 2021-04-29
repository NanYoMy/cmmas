import logging
import sys
import os,shutil

import numpy as np

# from config.configer import reg_config


import logging

def getLogger(logger_name,config=None,level=logging.DEBUG):
    if not os.path.exists(config['Inference']['dir_save']):
        os.makedirs(config['Inference']['dir_save'])
    file_handler = logging.FileHandler(filename=config['Inference']['dir_save']+"/my-model"+'.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=level,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger(logger_name)
    return logger


def getLoggerV3(logger_name,dirname,level=logging.DEBUG):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file_handler = logging.FileHandler(filename=dirname+"/my-model.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=level,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger(logger_name)
    return logger


def print_result(arr):
    # log = getLogger("inference", reg_config)
    # nparr=np.array(arr)
    # log.info("mean:"+str(np.mean(nparr)))
    # log.info("std:"+str(np.std(nparr)))
    pass