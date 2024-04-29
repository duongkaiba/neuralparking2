import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import function.helper as helper

# Load model
# detect plate
yolo_LP_detect = torch.hub.load(r'yolov5', 'custom', path=r'model/LP_detector_nano_61.pt', force_reload=True, source='local')
# recognition with ocr
yolo_license_plate = torch.hub.load(r'yolov5', 'custom', path=r'model/LP_ocr_nano_62.pt', force_reload=True, source='local')

# set model confidence threshold 
yolo_license_plate.conf = 0.60

#enter image path here
img_file = "test_image/test13.png"
img = cv2.imread(img_file)
plates = yolo_LP_detect(img, size=640)

list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()
count = 0
if len(list_plates) == 0:
    lp = helper.read_plate(yolo_license_plate,img)
    if lp != "unknown":
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        count+=1
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    flag = 1
                    break
            if flag == 1:
                break
            
print(list_read_plates)
cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()