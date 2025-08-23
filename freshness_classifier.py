# freshness_classifier.py
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import os
import cv2  

class FreshnessClassifier:
    def __init__(self, 
                 model_path="models/resnet_model.keras", 
                 encoder_path="models/label_encoder.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        self.le = joblib.load(encoder_path)
        print(f"✅ Label encoder loaded: {list(self.le.classes_)}")
    
    def classify(self, image):
        # Convert OpenCV to PIL
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        pil_image = Image.fromarray(image)
        
        # Preprocess
        img = pil_image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred_prob = self.model.predict(img_array, verbose=0)[0][0]
        pred_class = 1 if pred_prob > 0.5 else 0
        prediction = self.le.inverse_transform([pred_class])[0]
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        
        return prediction, confidence