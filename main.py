import os
# import shutil
import xml.etree.ElementTree as ET
import cv2 as cv2
import numpy as np
from tqdm import tqdm, trange
# from data_aug.data_aug import *
# from data_aug.bbox_util import *
# from Darknet import darknet
import time
from utils import CheckSavePath

classes = []
# classes = ["A", "B", "C", "D", "E"]

def GetXMLTree(file_path):
    tree, root = None, None
    with open(file_path, encoding="utf8") as f:
        tree = ET.parse(f)
        root = tree.getroot()
    return tree, root

def convert(size, box):
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

def GetLabel(root, convert=True):
    label_list = []
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = str(obj.find('name').text)
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        bb = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        if convert:
            bb = convert((w,h), bb)
        print({"label": label, "bndbox": bb})
        label_list.append({"label": label, "bndbox": bb})
    return label_list

def SliceImage(image, bnbbox):
    temp = image[int(round(bnbbox[1])):int(round(bnbbox[3])), int(round(bnbbox[0])):int(round(bnbbox[2]))]
    # ShowImage(temp)
    # temp = modify_contrast_and_brightness2(temp, brightness=40, contrast=100)
    return temp

def AssignImageRegion(image_path, coordinates_list, resized=None):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    time.sleep(0.1)
    images = []
    for label in coordinates_list:
        img = SliceImage(image, label['bndbox'])
        if resized:
            img  = cv2.resize(img, resized)
        label['image'] = img
    return coordinates_list

def SaveImageandInfo(src_path, label_list, saveDir):
    prefix = src_path.split(os.sep)[-1].split('.')[0]
    fmt = src_path.split(os.sep)[-1].split('.')[-1]
    for i, item in enumerate(label_list):
        save_dir = os.path.join(saveDir, item['label'])
        save_dir = CheckSavePath(save_dir)
        save_path = "{}/{}.{}".format(save_dir, prefix+'-'+str(i), fmt)
        # ShowImage(item['image'], names='out', wait=1000)
        cv2.imwrite(save_path, item['image'])
        time.sleep(0.1)

def ClassColors(classes= ["A", "B", "C", "D", "E"]):
    import random
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in classes}

def DrawImages(detections, image, class_colors, names='out', wait = 0):
    from Darknet import darknet
    draw_image = darknet.draw_boxes(detections, image.copy(), class_colors)
    ShowImage(draw_image, names=names, wait = wait)
    pass

def ShowImage(image, names='out', wait = 0):
    cv2.imshow(names, image)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()


def main():
    global classes
    # classes = ["DarkInWhite", "Edge", "WhiteInDark"]
    classes = ["A", "B", "C", "D", "E"]
    selectLabel = 'A'
    annotations_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\images'
    images_dir_path = r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\images'
    save_dir = r'D:\Users\Kent Tsai\Documents\datasets\FS-0302\128x128'
    annotations_files_path = [file_path for file_path in os.listdir(annotations_dir_path) if file_path.endswith('.xml')]
    images_file_path = [image_path for image_path in os.listdir(images_dir_path) if image_path.endswith('.bmp')]
    

    assert len(annotations_files_path) == len(images_file_path), "len(annotations_files_path) != len(images_file_path)"

    # Split defects of images for classification
    label_list = list()
    for i, (file_path, image_path) in enumerate(zip(annotations_files_path, images_file_path)):
        try:
            print("{}, {}\t{}".format(i, file_path, image_path))
            file_path = os.path.join(annotations_dir_path, file_path)
            image_path = os.path.join(images_dir_path, image_path)
            tree, root = GetXMLTree(file_path)
            label_list = GetLabel(root, False)
            label_list = AssignImageRegion(image_path, label_list, resized=(128,128))
            SaveImageandInfo(image_path, label_list, save_dir)
        except Exception as ex:
            print("Error: {}".format(ex))
            print("Images: {}".format(image_path))

   
    return 0


if __name__=='__main__':
    main()