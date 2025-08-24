import cv2
import torch
from ultralytics import YOLO
import os

class ChickenPartDetector:
    def __init__(self, weights_path="models/yolo11m.pt"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weight file tidak ditemukan: {weights_path}")
        
        try:
            self.model = YOLO(weights_path)
            self.model.to('cpu')  # Pastikan pakai CPU
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            raise Exception(f"Gagal memuat YOLO model: {str(e)}")
            
        self.class_names = ["breast", "leg", "quarter", "thigh", "wing"]
    
    def detect(self, image, conf_threshold=0.5):
        """
        Deteksi part ayam dalam gambar
        """
        try:
            # Deteksi dengan YOLO
            results = self.model(image, batch=1, conf=conf_threshold, imgsz=320, verbose=False)
            result = results[0]
            boxes = result.boxes
            
            detections = []
            img_with_boxes = image.copy()
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Ambil koordinat bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Pastikan cls_id valid
                    if cls_id < len(self.class_names):
                        label = self.class_names[cls_id]
                    else:
                        label = f"Part-{cls_id}"
                    
                    # Gambar bounding box
                    color = (0, 255, 0)  # Hijau
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    
                    # Tambahkan label
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(img_with_boxes, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Simpan deteksi
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': label,
                        'confidence': conf,
                        'class_id': cls_id
                    })
            
            return detections, img_with_boxes
            
        except Exception as e:
            raise Exception(f"Detection error: {str(e)}")
