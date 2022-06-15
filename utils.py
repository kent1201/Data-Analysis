#-*- coding : utf-8-*-
# coding:unicode_escape
import os
import re
import sys
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import argparse
import cv2 as cv
import numpy as np
import configparser
# from apscheduler.schedulers.blocking import BlockingScheduler
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

def ArgParse():
    parser = argparse.ArgumentParser(description='Please input your setting')
    parser.add_argument('--annotations_dir_path', type=str, default=r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\images')
    parser.add_argument('--images_dir_path', type=str, default=r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\images')
    parser.add_argument('--save_images_dir', type=str, default=r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\128x128')
    parser.add_argument('--save_annotation_dir', type=str, default=r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\128x128')
    return parser

def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            raise RuntimeError("Create dir error: {}".format(ex))
    return dir_path

def CreateLog():

    config = configparser.ConfigParser()
    fp_dir = os.getcwd()
    config_path = os.path.join(fp_dir, 'Configs', 'logger.ini')
    config.read(config_path)

    # Get Config
    log_dir_path = config.get('default', 'log_dir_path')
    log_name = config.get('default', 'log_name')
    level = config.get('default', 'level')
    record_mode = config.get('default', 'record_mode')
    
    # Check and Create directories by path
    log_filename = log_name
    log_dir_path = CheckSavePath(log_dir_path)
    
    # Create file
    if is_empty(log_filename):    
        log_filename = "{}_AI_Inference".format(os.getpid())
    else:
        log_filename = "{}_{}".format(os.getpid(), log_filename)
    log_file_path = os.path.join(log_dir_path, log_filename)
    
    # Create logger
    logger = logging.getLogger()
    
    # Set Level
    if level == "info":
        logger.setLevel(logging.INFO)
    elif level == "debug":
        logger.setLevel(logging.DEBUG)
    elif level == "error":
        logger.setLevel(logging.ERROR)
    
    # Set handler
    file_handler = None
    if record_mode == "Time":
        when = config.get('Time', 'when')
        interval = config.getint('Time', 'interval')
        Time_backupCount = config.getint('Time', 'backupcount')
        file_handler = TimedRotatingFileHandler(
            filename=log_file_path, when=when, interval=interval, backupCount=Time_backupCount
        )
    elif record_mode == "Size":
        # 3 MB
        Size = config.getint('Size', 'size')
        Size_BackupCount = config.getint('Size', 'backupcount')
        file_handler = RotatingFileHandler(
            filename=log_file_path, mode='a', maxBytes=Size*1024*1024, backupCount=Size_BackupCount, encoding='utf-8', delay=0
        )
    file_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.log$")
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )
    
    stream_handler = logging.StreamHandler(sys.stdout)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# def CreateLog(log_path=""):
#     log_filename = log_path
#     if is_empty(log_path):
#         # 通過下面的方式進行簡單配置輸出方式與日誌級別
#         log_dir_path = CheckSavePath('./Logs')
#         log_filename = os.path.join(log_dir_path, "{}.log".format(datetime.datetime.now().strftime("%Y-%m-%d")))
#     try:
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)])
#     except Exception as ex:
#         print("[Create log] {}".format(ex))   
#     # logging.debug('debug message')
#     # logging.info('info message')
#     # logging.warn('warn message')
#     # logging.error('error message')
#     # logging.critical('critical message')
#     return logging

def ExtractZip(zip_path, output_path=""):
    import zipfile
    if not output_path:
        output_path = os.path.dirname(zip_path)
    else:
        CheckSavePath(output_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_path)
    return output_path

        
def DeletFile(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            # os.remove() function to remove the file
            os.remove(file_path)
        except Exception as ex:
            print("[DeleteFile] Delete file: {} failed. error: {}".format(file_path, ex))




def String2Tuple(input_str):
    return tuple(map(int, input_str[1:-1].split(', ')))

def is_empty(any_structure):
    if any_structure:
        # print('Structure is not empty.')
        return False
    else:
        # print('Structure is empty.')
        return True

def GetNowTime_yyyymmddhhMMss():
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return now_time    

def CreateExampleImage(image_shape, batch_size):
    images_list = []
    for _ in range(0, batch_size):
        blank_image = np.ones(image_shape, np.uint8)
        if image_shape[2] == 3:
            blank_image = cv.cvtColor(blank_image, cv.COLOR_BGR2RGB)
        images_list.append({'image': blank_image})
    return images_list

# def GetScheduler(schedulerName='BlockingScheduler', executorName='ProcessPoolExecutor', executerCounts=5):
#     executor = None
#     if executorName == 'ProcessPoolExecutor':
#         executor = {'default': ProcessPoolExecutor(executerCounts)}
#     elif executorName == 'ThreadPoolExecutor':
#         executor = {'default': ThreadPoolExecutor(executerCounts)}
#     else:
#         raise ValueError("{} Still not supported.".format(executorName))
#     if schedulerName == 'BlockingScheduler':
#         return BlockingScheduler(executors=executor)
#     elif  schedulerName == 'AsyncIOScheduler':
#         return AsyncIOScheduler(executors=executor)
#     elif schedulerName == 'BackgroundScheduler':
#         return BackgroundScheduler(executors=executor)
#     else:
#         raise ValueError("{} Still not supported.".format(schedulerName))


if __name__=='__main__':
    image = CreateExampleImage((416, 416, 3), 1)
    cv.imshow("Simple_black", image[0])
    # 按下任意鍵則關閉所有視窗
    cv.waitKey(0)
    cv.destroyAllWindows()