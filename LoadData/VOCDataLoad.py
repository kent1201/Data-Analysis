import os
import xml.etree.ElementTree as ET
import cv2 as cv2
import numpy as np
import time
from tqdm import tqdm

def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            raise RuntimeError("Create dir error: {}".format(ex))
    return dir_path


class VOCDataLoader:
    def __init__(self, logger):
        self.logger = logger
        self.__Reset__()
    
    def __Reset__(self):
        self.logger.info("Reset VOC data loader.")
        self.image_dir = None
        self.annotation_dir = None
        self.images_file_path = None
        self.annotations_files_path = None
        self.data_list = list()
    
    def LoadData(self, image_dir, annotation_dir):
        try:
            self.image_dir, self.annotation_dir = image_dir, annotation_dir
            self.annotations_files_path = [file_path for file_path in os.listdir(annotation_dir) if file_path.endswith('.xml')]
            self.images_file_path = [image_path for image_path in os.listdir(image_dir) if os.path.splitext(image_path)[1] in ['.bmp', '.jpg', '.png']]
        except Exception as ex:
            self.logger.error("[VOCDataLoader-LoadData] Load Data Error {}".format(ex))
            raise ValueError("[VOCDataLoader-LoadData] Load Data Error {}".format(ex))
        
        if len(self.annotations_files_path) != len(self.images_file_path):
            self.logger.error("[VOCDataLoader-LoadData] The number of the annotation files is not equal to the number of the image files.")
            raise ValueError("[VOCDataLoader-LoadData] The number of the annotation files is not equal to the number of the image files.")
        
    def DataTransform(self, resize=(32, 32), save_dir=""):
        # Split defects of images for classification
        self.logger.info("[VOCDataLoader-DataTransform] Start Spliting objects from images...")
        for file_path, image_path in tqdm(zip(self.annotations_files_path, self.images_file_path), total=len(self.annotations_files_path)):
            try:
                label_list = list()
                file_path = os.path.join(self.annotation_dir, file_path)
                image_path = os.path.join(self.image_dir, image_path)
                tree, root = self.__GetXMLTree(file_path)
                label_list = self.__GetLabel(root, False)
                label_list = self.__AssignImageRegion(image_path, label_list, resized=resize)
                if save_dir:
                    self.__SaveImageandInfo(image_path, label_list, save_dir)
                self.data_list.extend(label_list)
            except Exception as ex:
                self.logger.error("[VOCDataLoader-DataTransform] Transform image {} failed: {}".format(image_path, ex))
        self.logger.info("[VOCDataLoader-DataTransform] Split objects from images end.")
    
    def GetData(self):
        return self.data_list
    
    def GetNumberofObjects(self):
        labels_count_dict = dict()
        try:
            for label_item in self.data_list:
                if not label_item["label"] in list(labels_count_dict.keys()):
                    labels_count_dict[label_item["label"]] = 1
                else:
                    labels_count_dict[label_item["label"]] += 1
        except Exception as ex:
            self.logger.error("[VOCDataLoader-GetNumberofObjects] Count labels failed: {}".format(ex))
            raise RuntimeError("[VOCDataLoader-GetNumberofObjects] Count labels failed: {}".format(ex))
        return labels_count_dict

    def __SaveImageandInfo(self, image_path, label_list, save_dir):
        prefix = image_path.split(os.sep)[-1].split('.')[0]
        fmt = image_path.split(os.sep)[-1].split('.')[-1]
        for i, item in enumerate(label_list):
            temp_dir = os.path.join(save_dir, item['label'])
            temp_dir = CheckSavePath(temp_dir)
            save_path = "{}/{}.{}".format(temp_dir, prefix+'-'+str(i), fmt)
            cv2.imwrite(save_path, item['image'])
            time.sleep(0.1)
    
    def __GetXMLTree(self, file_path):
        tree, root = None, None
        with open(file_path, encoding="utf8") as f:
            tree = ET.parse(f)
            root = tree.getroot()
        return tree, root
    
    def __GetLabel(self, root, convert=True):
        label_list = []
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            label = str(obj.find('name').text)
            xmlbox = obj.find('bndbox')
            bb = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            if convert:
                bb = self.__CoordinateConvert((w,h), bb)
            # print({"label": label, "bndbox": bb})
            label_list.append({"label": label, "bndbox": bb})
        return label_list
    
    def __AssignImageRegion(self, image_path, coordinates_list, resized=None):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)
        time.sleep(0.1)
        for label in coordinates_list:
            img = self.__SliceImage(image, label['bndbox'])
            if resized:
                img  = cv2.resize(img, resized)
            label['image'] = img
        return coordinates_list
    
    def __SliceImage(self, image, bnbbox):
        temp = image[int(round(bnbbox[1])):int(round(bnbbox[3])), int(round(bnbbox[0])):int(round(bnbbox[2]))]
        # ShowImage(temp)
        # temp = modify_contrast_and_brightness2(temp, brightness=40, contrast=100)
        return temp
    
    def __CoordinateConvert(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)


if __name__=='__main__':
    annotations_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS_0609_model2\train'
    images_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS_0609_model2\train'
    # voc_dataloader = VOCDataLoader(logger)
    # voc_dataloader.LoadData(images_dir_path, annotations_dir_path)
    # voc_dataloader.DataTransform(resize=(32, 32), save_dir="./")
    # data_list = voc_dataloader.GetData()
    # data_count = voc_dataloader.GetNumberofObjects()
    # for data in data_list:
    #     print(data)
    # print(data_count)
    