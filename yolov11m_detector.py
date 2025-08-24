# yolov11m_detector.py
import cv2
from ultralytics import YOLO
import os

class ChickenPartDetector:
    def __init__(self, weights_path="models/yolo11m.pt"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weight file tidak ditemukan: {weights_path}")
        
        # Load model
        self.model = YOLO(weights_path)
        self.model.to("cpu")
        
        # Nama class part ayam
        self.class_names = ["Breast", "Leg", "Quarter", "Thigh", "Wing"]
    
    def detect(self, image, conf_threshold=0.5):
        """
        Deteksi part ayam dalam gambar

        Args:
            image: Gambar input (numpy array)
            conf_threshold: Threshold confidence (default: 0.5)

        Returns:
            list: Daftar deteksi dengan bbox, label, confidence
            numpy.ndarray: Gambar dengan bounding box
        """
        results = self.model(image, batch=1, conf=conf_threshold, imgsz=320, verbose=False)
        result = results[0]
        boxes = result.boxes

        detections = []
        img_with_boxes = image.copy()

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Pastikan cls_id valid
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Part-{cls_id}"

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,          # Part ayam
                    "confidence": conf,
                    "class_id": cls_id
                })

        return detections, img_with_boxes
