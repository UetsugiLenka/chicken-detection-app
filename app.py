# app.py (Full Hugging Face Version - Label Fix)
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🐔 Deteksi & Klasifikasi Daging Ayam",
    page_icon="🐔",
    layout="wide"
)


import cv2
import numpy as np
from PIL import Image
import os
import time
from huggingface_hub import hf_hub_download  

# --- DOWNLOAD MODEL OTOMATIS DARI HUGGING FACE ---
@st.cache_resource
def download_models():
    """Download model dari Hugging Face jika belum ada"""
    
    # Buat folder models
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Download YOLOv11m dari Hugging Face
    yolo_path = "models/yolo11m.pt"
    if not os.path.exists(yolo_path):
        # st.info("📥 Downloading YOLOv11m model from Hugging Face...")  # ← dihapus
        try:
            hf_hub_download(
                repo_id="UetsugiLenka/chicken-models",
                filename="yolo11m.pt",
                revision="main",
                local_dir="models"
            )
            # st.success("✅ YOLOv11m downloaded successfully!")  # ← dihapus
        except Exception as e:
            st.error(f"❌ Gagal download YOLOv11m: {e}")
    
    # Download ResNet50 dari Hugging Face
    resnet_path = "models/resnet_model.keras"
    if not os.path.exists(resnet_path):
        # st.info("📥 Downloading ResNet50 model from Hugging Face...")  # ← dihapus
        try:
            hf_hub_download(
                repo_id="UetsugiLenka/chicken-models",
                filename="resnet_model.keras",
                revision="main",
                local_dir="models"
            )
            # st.success("✅ ResNet50 downloaded successfully!")  # ← dihapus
        except Exception as e:
            st.error(f"❌ Gagal download ResNet50: {e}")
        
# Jalankan download model
download_models()

# --- IMPORT MODEL MODULES ---
try:
    from yolov11m_detector import ChickenPartDetector
    from freshness_classifier import FreshnessClassifier
    print("✅ Modul berhasil diimport")
except Exception as e:
    st.error(f"❌ Error importing modules: {e}")
    st.stop()

# --- TITLE ---
st.title("🐔 Deteksi & Klasifikasi Daging Ayam")
st.markdown("---")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    """Load YOLOv11m and ResNet50 models"""
    try:
        detector = ChickenPartDetector(weights_path="models/yolo11m.pt")
        classifier = FreshnessClassifier(model_path="models/resnet_model.keras")
        return detector, classifier
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

with st.spinner("🔄 Memuat model..."):
    detector, classifier = load_models()

if detector is None or classifier is None:
    st.stop()

st.success("✅ Model berhasil dimuat!")

# --- SIDEBAR ---
st.sidebar.header("📷 Pilihan Input")

# Deteksi lokal/cloud
import socket
try:
    is_local = "localhost" in socket.gethostname() or "127.0.0.1" in socket.gethostbyname(socket.gethostname())
except:
    is_local = False

if is_local:
    input_option = st.sidebar.radio(
        "Pilih sumber gambar:",
        ("Upload Gambar", "Kamera Live")
    )
else:
    input_option = st.sidebar.radio(
        "Pilih sumber gambar:",
        ("Upload Gambar",)
    )
    st.sidebar.info("ℹ️ Kamera Live hanya tersedia di lokal")

# --- CONFIDENCE THRESHOLD SLIDER ---
st.sidebar.header("⚙️ Pengaturan Deteksi")
confidence_threshold = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info("""
**_legend_:**
- 🟢 **Hijau**: Segar
- 🔴 **Merah**: Busuk
""")

# --- MAIN CONTENT ---
if input_option == "Upload Gambar":
    st.subheader("📤 Upload Gambar Ayam")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diupload", use_container_width=True)

    # Konversi ke OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Deteksi part ayam
    with st.spinner("🔍 Mendeteksi part ayam..."):
        start_time = time.time()
        detections, img_bgr = detector.detect(image_cv, conf_threshold=confidence_threshold)
        deteksi_time = time.time() - start_time

    # Klasifikasi kesegaran
    results = []
    if detections:
        with st.spinner("🧠 Mengklasifikasi kesegaran..."):
            klasifikasi_start = time.time()
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det['bbox'])
                crop = image_cv[y1:y2, x1:x2]
                
                if crop.size > 0:
                    try:
                        pred, conf = classifier.classify(crop)
                        results.append({
                            'part': det['label'],
                            'freshness': pred,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })
                    except Exception as e:
                        st.error(f"❌ Error klasifikasi part {i+1} ({det['label']}): {str(e)}")
                        results.append({
                            'part': det['label'],
                            'freshness': 'Error',
                            'confidence': 0.0,
                            'bbox': (x1, y1, x2, y2)
                        })
            klasifikasi_time = time.time() - klasifikasi_start

    # 🔍 Filter overlap (manual NMS)
    filtered_results = []
    used_boxes = []

    for result in results:
        x1, y1, x2, y2 = result['bbox']
        area = (x2 - x1) * (y2 - y1)
        
        overlap = False
        for used in used_boxes:
            ux1, uy1, ux2, uy2 = used
            ix1, iy1 = max(x1, ux1), max(y1, uy1)
            ix2, iy2 = min(x2, ux2), min(y2, uy2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = area + (ux2 - ux1) * (uy2 - uy1) - intersection
                iou = intersection / union if union > 0 else 0
                if iou > 0.5:
                    overlap = True
                    break
        
        if not overlap:
            filtered_results.append(result)
            used_boxes.append((x1, y1, x2, y2))

    results = filtered_results

    # Gambar bounding box & label
    img_with_boxes = img_bgr.copy()
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        freshness = result['freshness']
        
        color = (0, 255, 0) if freshness.lower() == 'segar' else (0, 0, 255)
        
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # --- Label diperbaiki ---
        label = f"{result['part']} - {freshness} ({result['confidence']:.2f})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x, label_y = x1, y1 - 10
        if label_y < 10:
            label_y = y1 + text_size[1] + 10
        cv2.rectangle(img_with_boxes, 
                     (label_x, label_y - text_size[1] - 5), 
                     (label_x + text_size[0], label_y + 5), 
                     (0, 0, 0), -1)
        cv2.putText(img_with_boxes, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Hasil Deteksi & Klasifikasi", use_container_width=True)

    # Tabel hasil
    if results:
        import pandas as pd
        df_results = pd.DataFrame(results)
        df_results = df_results[['part', 'freshness', 'confidence']]
        df_results.columns = ['Part Ayam', 'Kesegaran', 'Confidence']
        st.dataframe(df_results, use_container_width=True)

         # Statistik (case-insensitive)
        segar_count = len([r for r in results if r['freshness'].lower() == 'segar'])
        busuk_count = len([r for r in results if r['freshness'].lower() == 'busuk'])
        
        st.markdown(f"""
        📊 **Statistik:**
        - 🟢 **Segar**: {segar_count} part
        - 🔴 **Busuk**: {busuk_count} part
        - ⏱️ **Waktu Deteksi**: {deteksi_time:.2f} detik
        - ⏱️ **Waktu Klasifikasi**: {klasifikasi_time:.2f} detik
        """)
    else:
        st.warning("❌ Tidak ada part ayam terdeteksi (confidence < threshold)")
        
# --- KAMERA LIVE ---
elif input_option == "Kamera Live":
    if not is_local:
        st.warning("🚫 Fitur kamera hanya tersedia di lokal.")
    else:
        st.subheader("📹 Kamera Live - Deteksi Real-Time")
        st.info("💡 Izinkan akses kamera di browser. Tekan tombol 'Start'.")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Gagal membuka kamera")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections, img_bgr = detector.detect(frame, conf_threshold=confidence_threshold)
            results = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    try:
                        pred, conf = classifier.classify(crop)
                        results.append({
                            'part': det['label'],
                            'freshness': pred,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })
                    except:
                        pass
            
            img_with_boxes = img_bgr.copy()
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                freshness = result['freshness']
                color = (0, 255, 0) if freshness.lower() == 'segar' else (0, 0, 255)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # --- Label diperbaiki ---
                label = f"{result['part']} - {freshness} ({result['confidence']:.2f})"
                cv2.putText(img_with_boxes, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(img_rgb, channels="RGB", caption="Live Detection")
            time.sleep(0.03)

        cap.release()
        FRAME_WINDOW.empty()

# --- FOOTER ---
st.markdown("---")
st.caption("🐔 Deteksi & Klasifikasi Daging Ayam - Skripsi 2025")










