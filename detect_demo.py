#!/usr/bin/env python
# Filename: detect_demo.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 January, 2023
"""

import os,sys
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, os.path.join(code_dir,'yolo'))       # for utils
# import ultralytics.yolo.utils as utils

from ultralytics import YOLO
import cv2



def train():
    model = YOLO("yolov8n.pt")  # pass any model type
    # model.train(data="coco128.yaml", epochs=3)
    model.val(data="coco128.yaml")

def validation():
    model = YOLO("yolov8n.yaml")
    # model.train(data="coco128.yaml", epochs=3)
    model.val()  # It'll automatically evaluate the data you trained.

def prediction():
    img = '000000000595.jpg'
    # model = YOLO("yolov8n.pt")
    model = YOLO("/Users/huanglingcao/codes/PycharmProjects/yghlc_ultralytics/runs/detect/train7/weights/best.pt")
    # from ndarray
    im2 = cv2.imread(img)
    results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels


def main():

    # train()
    # validation()

    prediction()



if __name__ == '__main__':
    main()