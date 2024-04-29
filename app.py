from flask import Flask, request, jsonify, render_template
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import numpy as np
import base64
import time
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)

CORS(app)

# Kết nối tới MongoDB Atlas
uri = "mongodb+srv://duonga1ne1:duong2003@cluster0.zjdjlqs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
 # Thay thế bằng URI thực tế của bạn từ MongoDB Atlas
client = MongoClient(uri)

# Lấy ra collection plate_number từ database neural_parking
db = client.neural_parking
collection = db.plate_number

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
# Load models from GitHub repository
yolo_LP_detect = torch.hub.load('duonguwu/yolov5', 'custom', path=r'model/LP_detector_nano_61.pt', source='github', force_reload=True, trust_repo=True)
yolo_license_plate = torch.hub.load('duonguwu/yolov5', 'custom', path=r'model/LP_ocr_nano_62.pt', source='github', force_reload=True, trust_repo=True)

# yolo_LP_detect = torch.hub.load(r'yolov5', 'custom', path=r'model/LP_detector_nano_61.pt', force_reload=True, source='local')
# yolo_license_plate = torch.hub.load(r'yolov5', 'custom', path=r'model/LP_ocr_nano_62.pt', force_reload=True, source='local')



yolo_license_plate.conf = 0.60  # Cấu hình ngưỡng tự tin

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({
        "message": "Xin chào Locket!!",
        "details": "Chi tiết vui lòng liên hệ: https://www.facebook.com/ngdw.04",
        "portfolio": "My portfolio: https://duongnguyen.onrender.com/"
    })


@app.route('/detect_license_plate', methods=['POST'])
def detect_license_plate():
    # Nhận ảnh từ request dưới dạng tệp
    image_file = request.files['image']

    # Đọc tệp ảnh và chuyển đổi nó thành mảng NumPy
    image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

    # Decode ảnh
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Giảm kích thước của ảnh mà không thay đổi tỷ lệ khung hình
    image_pil = Image.fromarray(img)
    max_size = (640, 640)
    image_pil.thumbnail(max_size, Image.LANCZOS)
    img = np.array(image_pil)

    # Tiếp tục xử lý ảnh như trước
    # Detect license plates
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()

    # Process license plates
    list_read_plates = set()
    image_with_boxes = img.copy()

    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0])  # xmin
            y = int(plate[1])  # ymin
            w = int(plate[2] - plate[0])  # xmax - xmin
            h = int(plate[3] - plate[1])  # ymax - ymin
            crop_img = img[y:y+h, x:x+w]
            lp = ""
            for cc in range(0, 2):
                for ct in range(0, 2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        flag = 1
                        break
                if flag == 1:
                    break

    
    # Trả về kết quả
    return jsonify({
        "license_plates": list(list_read_plates),
    })

# @app.route('/detect_license_plate2', methods=['POST'])
# def detect_license_plate2():
#     # Nhận ảnh từ request dưới dạng tệp
#     image_file = request.files['image']

#     # Đọc tệp ảnh và chuyển đổi nó thành mảng NumPy
#     image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

#     # Decode ảnh
#     img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

#     # In kích thước ảnh sau khi đọc
#     print(f"Image shape after reading: {img.shape}")

#     # Giảm kích thước của ảnh mà không thay đổi tỷ lệ khung hình
#     image_pil = Image.fromarray(img)
#     max_size = (640, 640)
#     image_pil.thumbnail(max_size, Image.LANCZOS)
#     img = np.array(image_pil)

#     # In kích thước ảnh sau khi resize
#     print(f"Image shape after resize: {img.shape}")

#     # In mức sử dụng bộ nhớ trước khi detect biển số
#     print(f"Memory usage before detection: {psutil.virtual_memory().used}")
#     a = psutil.virtual_memory().used
#     # Detect license plates
#     start_time = time.time()
#     plates = yolo_LP_detect(img, size=640)
#     list_plates = plates.pandas().xyxy[0].values.tolist()
#     detection_time = time.time() - start_time
#     print(f"Detection time: {detection_time}")

#     # Process license plates
#     list_read_plates = set()
#     image_with_boxes = img.copy()

#     if len(list_plates) == 0:
#         lp = helper.read_plate(yolo_license_plate, img)
#         print(f"OCR result: {lp}")
#         if lp != "unknown":
#             list_read_plates.add(lp)
#     else:
#         for plate in list_plates:
#             flag = 0
#             x = int(plate[0])  # xmin
#             y = int(plate[1])  # ymin
#             w = int(plate[2] - plate[0])  # xmax - xmin
#             h = int(plate[3] - plate[1])  # ymax - ymin
#             crop_img = img[y:y+h, x:x+w]

#             total_ocr_time = 0
#             for cc in range(0, 2):
#                 for ct in range(0, 2):
#                     start_ocr_time = time.time()
#                     lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
#                     ocr_time = time.time() - start_ocr_time
#                     total_ocr_time += ocr_time
#                     print(f"OCR result (x={x}, y={y}, w={w}, h={h}, cc={cc}, ct={ct}): {lp}, Time: {ocr_time}")
#                     if lp != "unknown":
#                         list_read_plates.add(lp)
#                         flag = 1
#                         break
#                 if flag == 1:
#                     break

#             # In tổng thời gian xử lý OCR
#         print(f"Total OCR time: {total_ocr_time}")
#         # Vẽ bounding box lên ảnh
#         cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 # Vẽ biển số lên ảnh
#         cv2.putText(image_with_boxes, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     # Tạo tên tệp ảnh tạm thời và lưu ảnh đã vẽ vào đó
#     temp_image_path = "temp_image.jpg"
#     cv2.imwrite(temp_image_path, image_with_boxes)
#     # # Tải ảnh đã vẽ lên Cloudinary
#     cloudinary_response = cloudinary.uploader.upload(temp_image_path)

#     # In mức sử dụng bộ nhớ sau khi detect biển số
#     print(f"Memory usage after detection: {psutil.virtual_memory().used}")
#     b = psutil.virtual_memory().used
#     # Xóa tệp ảnh tạm thời

#     # Trả về kết quả dưới dạng JSON
#     response_data = {
#         "license_plates": list(list_read_plates),
#         "image_url": cloudinary_response['secure_url'],
#         "detection_time": detection_time,
#         "total_ocr_time": total_ocr_time,
#         "memory_usage_before_detection": psutil.virtual_memory().used,
#         "memory_usage_after_detection": b - a
#     }

#     return jsonify(response_data)





# recognized_plates = {}

# @socketio.on('process_video', namespace='/video')
# def process_video():
#     video_path = "test_image/plate.mp4"  # Đường dẫn tới video trong dự án Flask của bạn
#     vid = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = vid.read()
#         if not ret:
#             break
#         process_frame(frame)
#     emit('video_processed', {'message': 'Video processed successfully'})

# def process_frame(frame):
#     plates = yolo_LP_detect(frame, size=640)
#     list_plates = plates.pandas().xyxy[0].values.tolist()
#     for plate in list_plates:
#         x = int(plate[0])
#         y = int(plate[1])
#         w = int(plate[2] - plate[0])
#         h = int(plate[3] - plate[1])
#         crop_img = frame[y:y+h, x:x+w]
#         rc_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
#         lp = ""
#         for cc in range(0, 2):
#             for ct in range(0, 2):
#                 lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
#                 if lp != "unknown" and lp not in recognized_plates:
#                     recognized_plates[lp] = time.time()
#                     # In ra biển số nhận dạng
#                     print("Detected plate:", lp)
#                     # Truy vấn biển số trong MongoDB Atlas và gửi response cho frontend
#                     process_detected_plate(lp)

# def process_detected_plate(plate):
#     # Truy vấn biển số trong MongoDB Atlas
#     result = collection.find_one({"number_plate": plate})
#     if result:
#         print("Processing plate:", plate)
#         # Gửi response đến frontend với biển số và thời gian nhận dạng được
#         socketio.emit('plate_detected', {'plate': plate, 'timestamp': time.time()}, namespace='/video')


if __name__ == '__main__':
    app.run(debug=True)
