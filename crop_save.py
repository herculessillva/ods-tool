from ctypes import *
import random
import os
import cv2
import time
import copy
from random import randint
import darknet.darknet as darknet
import glob
from natsort import natsorted


def convertBack( x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
    
def list_to_dict(list_yolo):
    result_dict = [None]*len(list_yolo)
    result_temp = {}
    for i, item in enumerate(list_yolo):
        x_item, y_item, w_item, h_item = item[2][0], item[2][1], item[2][2], item[2][3]
        result_temp['label'] = item[0]
        result_temp['confidence'] = item[1]
        tlx, tly, brx, bry = convertBack(
            float(x_item), float(y_item), float(w_item), float(h_item))
        result_temp['topleft'] = {}
        result_temp['topleft']['x'] = tlx
        result_temp['topleft']['y'] = tly
        result_temp['bottomright'] = {}
        result_temp['bottomright']['y'] = bry
        result_temp['bottomright']['x'] = brx
        result_dict[i] = copy.deepcopy(result_temp)
    return result_dict


'''------------------------------------------------------------------------------------------------'''
#Detecção
threshold = .5
weights = './YOLO_files/yolov4-obj_final.weights'
netcfg  = './YOLO_files/yolov4-obj.cfg'
data = './YOLO_files/obj.data'
in_path = '/home/hercules/Downloads/DS_13deMaio_DVR/Frames/'
OUT_PATH = '/home/hercules/Documents/codes/tests_videos_atlanta/out_DS_13deMaio_DVR/'
dsize = (1920,1080)

'''------------------------------------------------------------------------------------------------'''

network, class_names, class_colors = darknet.load_network(
            netcfg,
            data,
            weights,
        )
# width = darknet.network_width(network)
# height = darknet.network_height(network)
width = 1920
height = 1080
# darknet_image = darknet.make_image(width, height, 3)

list_imgs_paths = natsorted(glob.glob(in_path+'*.png'))

total = len(list_imgs_paths)
num_frame = 0;
for img_path in list_imgs_paths[:]:
    print('reading {}/{}'.format(num_frame+1,total))
    
    frame = cv2.imread(img_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
    
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=threshold)
    darknet.free_image(darknet_image)
    
    if len(detections):
        results = list_to_dict(detections)

        for i,result in enumerate(results):
            y_topleft = max(0, result['topleft']['y'])
            y_bottomright = min(result['bottomright']['y'], frame_resized.shape[0])
            x_topleft = max(0, result['topleft']['x'])
            x_bottomright = min(result['bottomright']['x'], frame_resized.shape[1])
            out_path = img_path.replace(in_path, OUT_PATH).replace('.png', '_{}.png'.format(str(i)))
            
            frame_write = frame_resized[y_topleft:y_bottomright, x_topleft:x_bottomright,:]

            # frame_write = cv2.resize(frame_resized, dsize)
            frame_write = cv2.cvtColor(frame_write, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(out_path, frame_write)
    num_frame += 1
