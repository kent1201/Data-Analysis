import os
import time
# import shutil
import xml.etree.ElementTree as ET
import cv2 as cv2
import numpy as np
from tqdm import tqdm, trange
from LoadData import VOCDataLoad, ClassificationDataLoad
from data_visualization import DataVisualization
from utils import CreateLog


def main():

    ## Create Logger
    logger = CreateLog()


    ## Load VOC data
    # annotations_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS_0609_model2\train'
    # images_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS_0609_model2\train'
    # voc_dataloader = VOCDataLoad.VOCDataLoader(logger)
    # voc_dataloader.LoadData(images_dir_path, annotations_dir_path)
    # voc_dataloader.DataTransform(resize=(32, 32))
    # data_list = voc_dataloader.GetData()
    # data_count = voc_dataloader.GetNumberofObjects()

    ## Load Classification data
    images_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS_0614_model2_defect\64x64'
    data_loader = ClassificationDataLoad.ClassificationDataLoader(logger=logger)
    data_loader.LoadData(images_dir_path)
    data_loader.DataTransform(resize=(32, 32))
    data_list = data_loader.GetData()
    data_count = data_loader.GetNumberofObjects()
    data_loader.SaveData("./test_image")
    
    ## Show label count of data
    for label, count in data_count.items():
        print("label: {}: {}".format(label, count))

    
    ## Data 2D/3D Visualization 
    # analysis_methods = 'tsne'    
    # logger.info("Select method: {}".format(analysis_methods))
    # data_visualization = DataVisualization(analysis=analysis_methods)
    # logger.info("Data Transforming...")
    # data_X, data_Y = data_visualization.dataTransform(data_list, gray_scale=False, norm=True)
    # logger.info("data_X: {}\n data_Y: {}".format(data_X[0], data_Y[0]))
    # logger.info("Visualization Start...")
    # data_visualization.visualization(data_X, data_Y, sampled_data=3000)
    

   
    return 0


if __name__=='__main__':
    main()