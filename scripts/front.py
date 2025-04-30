import os
import sys
import argparse
import glob
import time
from utils import *
from ocr import *
import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    default='../models/front_ncnn_model')
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default='1920x1080')
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
topic = 'Topic'
# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']


source_type = 'usb'
usb_idx = int(img_source[3:])


# Parse user-specified display resolution
resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])


if source_type == 'video': cap_arg = img_source
elif source_type == 'usb': cap_arg = usb_idx
cap = cv2.VideoCapture(cap_arg)


ret = cap.set(3, resW)
ret = cap.set(4, resH)


# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
states = {'Plaza_1':True,'Plaza_2':True,'Plaza_3':True}
# Begin inference loop

while True:
    #Falta actualizar states desde el servidor mqtt
    try:
        frame,detections=get_detections(cap,model)
    except:
        break

    spots,objects = process_detections(frame,detections,labels,min_thresh,states,bbox_colors)
    obj_count = len(spots) + len(objects)
    
    display_detections(frame,obj_count)
    
    process_front(frame,spots,objects,states,topic)

    if(get_controls(frame)):break
    
cap.release()
cv2.destroyAllWindows()
