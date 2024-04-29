import cv2
from ultralytics import YOLO
import torch
import math 
import function.utils_rotate as utils_rotate
import function.helper as helper
import time
from pymongo import MongoClient

# Load model
# detect plate
yolo_LP_detect = torch.hub.load(r'yolov5', 'custom', path=r'model/LP_detector_nano_61.pt', force_reload=True, source='local')
# recognition with ocr
yolo_license_plate = torch.hub.load(r'yolov5', 'custom', path=r'model/LP_ocr_nano_62.pt', force_reload=True, source='local')

# set model confidence threshold 
yolo_license_plate.conf = 0.60

recognized_plates = {}

# Kết nối tới MongoDB Atlas
uri = "mongodb+srv://duonga1ne1:duong2003@cluster0.zjdjlqs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
 # Thay thế bằng URI thực tế của bạn từ MongoDB Atlas
client = MongoClient(uri)

# Lấy ra collection plate_number từ database neural_parking
db = client.neural_parking
collection = db.plate_number

def process_frame(frame):
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    for plate in list_plates:
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin 
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = frame[y:y+h, x:x+w]
        rc_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown" and lp not in recognized_plates:
                    recognized_plates[lp] = time.time()
                    # In ra biển số nhận dạng
                    print("Detected plate:", lp)
                    # Truy vấn biển số trong MongoDB Atlas
                    process_detected_plate(lp)
                    
def process_detected_plate(plate):
    # Truy vấn biển số trong MongoDB Atlas
    result = collection.find_one({"number_plate": plate})
    if result:
        print("Processing plate:", plate)
        # Gửi response đến frontend
        send_response_to_frontend(plate)

def send_response_to_frontend(plate):
    # Gửi response đến frontend
    print("Sending response to frontend for plate:", plate)

def process_video():
    vid = cv2.VideoCapture("test_image/plate.mp4")
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        process_frame(frame)
        
        # Hiển thị video
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()

def get_recognized_plates():
    result = {}
    for plate, timestamp in recognized_plates.items():
        if time.time() - timestamp <= 5:  # Thời gian tối đa giữ lại là 5 giây
            result[plate] = timestamp
    return result

# Chạy hàm xử lý video
process_video()

# In ra các biển số đã nhận dạng được
print("Recognized Plates:")
print(get_recognized_plates())


