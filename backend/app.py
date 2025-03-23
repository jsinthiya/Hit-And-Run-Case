import os
import zipfile
import cv2
import numpy as np
import re
import uuid  # To generate unique folder names
from flask import Flask, request, jsonify
from ultralytics import YOLO
import shutil
from flask_cors import CORS
import easyocr

app = Flask(__name__)
CORS(app)

# Load the YOLOv8 model
model = YOLO("weight.pt")
names = model.model.names

# Load the YOLO model for license plate detection
license_plate_model = YOLO("license_plate_detector.pt")

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['bn'])

def extract_metadata_from_filename(filename):
    pattern = r"cctv(\d+)_(\d+\.\d+)-(\d+\.\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.search(pattern, filename)

    if match:
        return {
            "cctv": int(match.group(1)),
            "loc": {
                "latitude": float(match.group(2)),
                "longitude": float(match.group(3))
            },
            "time": match.group(4)
        }
    return {
        "cctv": None,
        "loc": {"latitude": None, "longitude": None},
        "time": None
    }

def is_target_color(class_name, car_color, car_number):
    """
    Checks if the detected class matches the target car color and number.
    """
    print(f"Checking {class_name} with {car_color} and {car_number}")

    if class_name is None:
        return False
    return class_name.lower() == car_color.lower()

@app.route('/detect_zip', methods=['POST'])
def detect_zip_images():
    if 'file' not in request.files:
        return jsonify({"error": "No file found in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Additional parameters from the request
    car_color = request.form.get('car_color', None)
    car_number = request.form.get('car_number', None)
    accident_latitude = request.form.get('accident_latitude', None)
    accident_longitude = request.form.get('accident_longitude', None)

    # Check if the uploaded file is a ZIP
    if not file.filename.endswith('.zip'):
        return jsonify({"error": "Please upload a ZIP file"}), 400

    # Create a unique folder for this ZIP file
    unique_folder = str(uuid.uuid4())  # Generate a unique folder name
    temp_dir = os.path.join("uploads", unique_folder)
    os.makedirs(temp_dir, exist_ok=True)

    # Extract ZIP contents
    zip_path = os.path.join(temp_dir, "uploaded.zip")
    file.save(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Folder to store detected images
    detected_dir = os.path.join(temp_dir, "detected")
    os.makedirs(detected_dir, exist_ok=True)

    all_detections = []

    for root, _, files in os.walk(temp_dir):  # Recursively go through all subdirectories
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, filename)

                # Extract metadata
                metadata = extract_metadata_from_filename(filename)

                image = cv2.imread(file_path)

                if image is None:
                    continue  # Skip unreadable images

                # Run YOLO object detection
                results = model.predict(image)

                detections = []
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.int().cpu().tolist()
                        class_ids = result.boxes.cls.int().cpu().tolist()
                        confidences = result.boxes.conf.float().cpu().tolist()

                        for box, class_id, conf in zip(boxes, class_ids, confidences):
                            class_name = names[class_id]
                            detection_info = {
                                "class": class_name,
                                "bbox": box,
                                "confidence": round(conf, 4)
                            }

                            if is_target_color(class_name, car_color, car_number):
                                print(f"Detected {class_name} with confidence {conf}")
                                # Crop the image based on the bounding box
                                x1, y1, x2, y2 = box
                                cropped_image = image[y1:y2, x1:x2]
                                
                                # Save the cropped image with metadata in the filename
                                cropped_filename = f"{class_name}_{x1}_{y1}_{x2}_{y2}_{filename}"
                                cropped_path = os.path.join(detected_dir, cropped_filename)
                                cv2.imwrite(cropped_path, cropped_image)

                                detection_info['cropped_image'] = cropped_filename

                            detections.append(detection_info)

                all_detections.append({
                    "filename": filename,
                    "metadata": metadata,
                    "detections": detections,
                    "car_color": car_color,
                    "car_number": car_number,
                    "accident_latitude": accident_latitude,
                    "accident_longitude": accident_longitude,
                })

    # Cleanup extracted files (ZIP)
    # shutil.rmtree(temp_dir, ignore_errors=True)

    return jsonify({
        "detection_id": unique_folder,
        "results": all_detections
        })

@app.route('/detect_license_plate', methods=['POST'])
def detect_license_plate():
    data = request.get_json()
    detection_id = data.get('detection_id')
    target_car_number = data.get('car_number')

    if not detection_id:
        return jsonify({"error": "detection_id is required"}), 400

    if not target_car_number:
        return jsonify({"error": "car_number is required"}), 400

    # Normalize the target car number by removing non-alphanumeric characters
    target_car_number = re.sub(r'\W+', '', target_car_number)

    temp_dir = os.path.join("uploads", detection_id)
    detected_dir = os.path.join(temp_dir, "detected")
    plate_dir = os.path.join(temp_dir, "plate")
    os.makedirs(plate_dir, exist_ok=True)

    all_plate_detections = []

    for root, _, files in os.walk(detected_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                if image is None:
                    continue  # Skip unreadable images

                # Run license plate detection
                results = license_plate_model.predict(image)

                for result in results:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        plate_img = image[y1:y2, x1:x2]

                        # Save cropped license plate image
                        plate_filename = f"plate_{x1}_{y1}_{x2}_{y2}_{filename}"
                        plate_path = os.path.join(plate_dir, plate_filename)
                        cv2.imwrite(plate_path, plate_img)

                        # Run OCR on the cropped license plate image
                        ocr_results = ocr_reader.readtext(plate_img)
                        car_number = None
                        if ocr_results:
                            car_number = ocr_results[0][-2]  # Extract the detected text
                            # Normalize the detected car number
                            car_number = re.sub(r'\W+', '', car_number)

                        # Calculate the number of matching characters
                        match_count = sum(1 for a, b in zip(car_number, target_car_number) if a == b) if car_number else 0

                        all_plate_detections.append({
                            "filename": filename,
                            "plate_image": plate_filename,
                            "bbox": [x1, y1, x2, y2],
                            "car_number": car_number,
                            "match_count": match_count,
                            "metadata": extract_metadata_from_filename(filename)
                        })

    # Sort detections by the number of matching characters in descending order
    all_plate_detections.sort(key=lambda x: x['match_count'], reverse=True)

    return jsonify({
        "detection_id": detection_id,
        "plate_detections": all_plate_detections
    })

if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
