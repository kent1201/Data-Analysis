import os
import time
import cv2 as cv2
from tqdm import tqdm
import numpy as np

def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            raise RuntimeError("Create dir error: {}".format(ex))
    return dir_path

class ClassificationDataLoader:
    def __init__(self, logger):
        self.logger = logger
        self.__Reset__()
    
    def __Reset__(self):
        self.logger.info("Reset classification data loader.")
        self.root_dir = None
        self.labels_dir = None
        self.data_list = list()
    
    def LoadData(self, root_dir):
        try:
            self.root_dir = root_dir
            self.labels_dir = [label_dir for label_dir in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, label_dir))]
            self.logger.debug("[ClassificationDataLoader-LoadData] labels_dir: {}".format(self.labels_dir))
        except Exception as ex:
            self.logger.error("[ClassificationDataLoader-LoadData] Load Data Error {}".format(ex))
            raise ValueError("[ClassificationDataLoader-LoadData] Load Data Error {}".format(ex))
        
    def DataTransform(self, resize=(32, 32)):
        # Read images for classification
        self.logger.info("[ClassificationDataLoader-DataTransform] Start transforming...")
        for label_dir in self.labels_dir:
            image_path = None
            try:
                self.logger.info("[ClassificationDataLoader-DataTransform] Start label {} transform.".format(label_dir))
                temp_label_dir = os.path.join(self.root_dir, label_dir)
                imgs_dir = os.listdir(temp_label_dir)
                for item in tqdm(imgs_dir):
                    if os.path.splitext(item)[1] in ['.bmp', '.jpg', '.png']:
                        image_path = os.path.join(self.root_dir, label_dir, item)
                        image = self.__GetImage(image_path, resize)
                        self.data_list.append({"image": image, "label": label_dir, "image_name": item})
            except Exception as ex:
                self.logger.error("[ClassificationDataLoader-DataTransform] Transform image {} failed: {}".format(image_path, ex))

    def SaveData(self, save_dir="./"):
        try:
            save_dir = CheckSavePath(save_dir)
            for data in self.data_list:
                data_dir = os.path.join(save_dir, data["label"])
                data_dir = CheckSavePath(data_dir)
                save_path = os.path.join(data_dir, data['image_name'])
                cv2.imwrite(save_path, data['image'])
        except Exception as ex:
            self.logger.error("[ClassificationDataLoader-SaveData] save image {} failed: {}".format(save_path, ex))
            raise RuntimeError("[ClassificationDataLoader-SaveData] Load Data Error {}".format(ex))
    
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
            self.logger.error("[ClassificationDataLoader-GetNumberofObjects] Count labels failed: {}".format(ex))
            raise RuntimeError("[ClassificationDataLoader-GetNumberofObjects] Count labels failed: {}".format(ex))
        return labels_count_dict
    
    def __GetImage(self, image_path, resize=None):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if resize:
            image  = cv2.resize(image, resize)
        if image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) < 3:
            image = np.expand_dims(image, 2)
        time.sleep(0.1)
        return image

