import os
import cv2
import numpy as np
import Algorithms.ADT07.darknet as darknet

"""
Here are ADT07 framework functions.
Now have Load model and one image detection function.
We expect to add batch_detection for multipli images in the next version
"""
# Load model with .cfg (_validation.cfg) and .weighs (_best.weights)

import configparser
config = configparser.ConfigParser()
fp_dir = os.getcwd()
config_path = os.path.join(fp_dir, 'Configs', 'algorithms_config.ini')
config.read(config_path)

def LoadModel(model_path, labels_list, model_info, logger):
    config_file, weights = '', ''
    for item in os.listdir(model_path):
        if item.endswith('_validation.cfg'):
            config_file = os.path.join(model_path, item)
        elif item.endswith('_best.weights'):
            weights = os.path.join(model_path, item)

    if not config_file:
        logger.error('0012: [ADT07] Cannot find .cfg file, please check your file.')
    if not weights:
        logger.error('0013: [ADT07] Cannot find .weight file, please check your file.')

    network, class_names, class_colors = darknet.load_network(
        config_file,
        labels_list,
        weights,
        batch_size=model_info['batch_size']
    )

    return network, class_names, class_colors

# Only one image detection
def ImageDetection(image, network, class_names, class_colors, model_info):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    image_resized = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=config.getfloat('M07', 'threshold'))
    darknet.free_image(darknet_image)
    # image = darknet.draw_boxes(detections, image_resized, class_colors)
    relative_detections = []
    for label, confidence, bbox in detections:
        bbox = convert2relative(image_resized, bbox)
        relative_detections.append((label, confidence, bbox))
    return relative_detections

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return (x/width, y/height, w/width, h/height)

def check_batch_shape(images, batch_size):
    # Image sizes should be the same width and height
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]

def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_resized = cv2.resize(image, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)

def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    # exception: access violation reading 0x0000017A91745040
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        # images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return batch_predictions