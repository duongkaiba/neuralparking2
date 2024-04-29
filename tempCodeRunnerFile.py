import torch
import cv2
from ultralytics import YOLO
import easyocr  # Nhập thư viện EasyOCR để đọc ký tự

model = YOLO('license_plate_detector.pt')

def detect_license_plate(imagepath = "./testimg/test30A04926.jpg"):
    try:
        img = cv2.imread(imagepath)

        # Chạy phát hiện với model YOLOv8
        results = model(imagepath)

        # Truy cập bounding box (sửa lỗi)
        for r in results:
            boxes = r.boxes.xyxy.tolist()
            if boxes:
                x_min, y_min, x_max, y_max = boxes[0]  # Bổ sung conf và clas
                width = x_max - x_min
                height = y_max - y_min
                return imagepath, int(x_min), int(y_min), int(width), int(height)
            else:
                raise ValueError("No license plate detected")

    except Exception as e:
        print(f"Error during detection: {e}")
    raise

### crop object
def crop(imagepath, x, y, w, h): 
    image = cv2.imread(imagepath)
    crop_img = image[y:y+h, x:x+w]
    path_crop = f"results_crop/{imagepath.split('/')[-1]}"
    cv2.imwrite(path_crop, crop_img)
    
    return path_crop

### extract value 
def OCR(path): 
    IMAGE_PATH = path
    reader = easyocr.Reader(['en'])
    result = reader.readtext(IMAGE_PATH)
    plate = ' '.join(detect[1] for detect in result)
    print("EXTRACT: ", plate)
    return

def main():
    try:
        path, x, y, w, h = detect_license_plate()
        cropped_img = crop(path, x, y, w, h)
        OCR(cropped_img)
    except ValueError:
        print("No detected plate")


if __name__ == '__main__':
  main()


