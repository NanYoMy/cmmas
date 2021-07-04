import logging
import sys
import os,shutil

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

from config.load_embedding_arg import level
def getLoggerV2(logger_name,args=None,level=level):
    if not os.path.exists(args.dir_save[0]):
        os.makedirs(args.dir_save[0])
    file_handler = logging.FileHandler(filename=args.dir_save[0]+"/my-model.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=level,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger(logger_name)
    return logger