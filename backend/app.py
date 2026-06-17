import os
import cv2
import numpy as np
import torch
import imghdr
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# ==============================================================================
# 1. KONFIGURASI KEAMANAN & AKSES (PRODUCTION HARDENING)
# ==============================================================================
# Pastikan NEXTJS_URL di set di env Render agar tidak default ke localhost
NEXTJS_URL = os.environ.get('NEXTJS_URL', 'http://localhost:3000')
CORS(app, origins=[NEXTJS_URL])

app.config['MAX_CONTENT_LENGTH'] = 35 * 1024 * 1024  # Maksimal file 35MB
limiter = Limiter(
    key_func=get_remote_address, 
    app=app, 
    default_limits=["300 per day"], 
    storage_uri="memory://"
)

# Rahasia bersama (Shared Secret) antara Vercel dan Render
API_KEY = os.environ.get("API_KEY")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/jpg', 'video/mp4', 'video/quicktime', 'video/x-msvideo'}

def allowed_file(filename, mime_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS and mime_type in ALLOWED_MIME_TYPES

def is_safe_image_content(file_stream):
    """Memeriksa struktur byte file untuk memastikan konten asli adalah gambar yang valid."""
    header = file_stream.read(2048)
    file_stream.seek(0)  # Kembalikan pointer pembacaan ke awal file
    file_type = imghdr.what(None, h=header)
    # Jika file merupakan video, imghdr mengembalikan None. 
    # Validasi ini krusial untuk memastikan endpoint image tidak disusupi skrip berbahaya.
    return file_type in ['jpeg', 'png', 'jpg'] if file_type else True

def require_api_key(f):
    """Decorator untuk memblokir request ilegal dari luar aplikasi Frontend Vercel Anda."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Jika API_KEY belum diset di env, lewatkan untuk kemudahan development lokal
        if API_KEY and request.headers.get("X-API-KEY") != API_KEY:
            return jsonify({"error": "Unauthorized access denied"}), 401
        return f(*args, **kwargs)
    return decorated

# ==============================================================================
# 2. INISIALISASI HARDWARE & MODEL YOLOv8 (.PT)
# ==============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_half = True if device == 'cuda' else False

# Path disesuaikan dengan arsitektur Docker /app WORKDIR di Render
MODEL_PATH = '/app/models/best.pt'
if not os.path.exists(MODEL_PATH):
    # Fallback ke path lokal jika dijalankan di luar lingkungan Docker produksi
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

try:
    print(f"🚀 Memuat model Ultralytics di {device.upper()} (Cascade Production Mode)...")
    model = YOLO(MODEL_PATH)
    
    # Warm-up model agar request pertama pengguna tidak terkena cloud timeout/lag
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_img, device=device, half=use_half, verbose=False)
    print("⚡ Model YOLOv8n siap memproses data.")
except Exception as e:
    print(f"CRITICAL: Gagal memuat model: {e}")
    model = None

# ==============================================================================
# 3. CORE LOGIC: CASCADE DETECTION PIPELINE
# ==============================================================================
def process_cascade(img, results_cars, is_tracking=False):
    cars = []
    plates = []
    
    if results_cars[0].boxes is not None:
        car_boxes = results_cars[0].boxes.cpu().numpy()
        
        for box in car_boxes:
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
            c_conf = float(box.conf[0])
            
            track_id = int(box.id[0]) if (is_tracking and box.id is not None) else None
            
            car_data = {
                "bbox": [cx1, cy1, cx2, cy2], 
                "conf": round(c_conf, 3), 
                "plate_detected": False
            }
            if track_id is not None: 
                car_data["track_id"] = track_id
                
            # --- TAHAP 2: CROP & DETEKSI PLAT NOMOR ---
            h, w = img.shape[:2]
            crop_y1, crop_y2 = max(0, cy1), min(h, cy2)
            crop_x1, crop_x2 = max(0, cx1), min(w, cx2)
            
            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                # Cari kelas 1 (Plat) khusus di dalam bounding box kendaraan hasil crop
                results_plates = model(crop_img, device=device, half=use_half, iou=0.4, conf=0.15, imgsz=320, classes=[1], verbose=False)
                
                if results_plates[0].boxes is not None:
                    for p_box in results_plates[0].boxes.cpu().numpy():
                        px1_rel, py1_rel, px2_rel, py2_rel = map(int, p_box.xyxy[0])
                        p_conf = float(p_box.conf[0])
                        
                        # Transformasi koordinat crop kembali ke koordinat gambar absolut utama
                        px1_abs = crop_x1 + px1_rel
                        py1_abs = crop_y1 + py1_rel
                        px2_abs = crop_x1 + px2_rel
                        py2_abs = crop_y1 + py2_rel
                        
                        plates.append({
                            "bbox": [px1_abs, py1_abs, px2_abs, py2_abs], 
                            "conf": round(p_conf, 3)
                        })
                        car_data['plate_detected'] = True
                        
            cars.append(car_data)
            
    return cars, plates

# ==============================================================================
# 4. ENDPOINTS API WITH HARDENED SECURITY
# ==============================================================================
@app.route('/api/detect', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def detect_image():
    if 'file' not in request.files:
        return jsonify({"error": "Data part file tidak ditemukan"}), 400
        
    file = request.files['file']
    
    if not allowed_file(file.filename, file.content_type):
        return jsonify({"error": "Ekstensi atau MIME tipe file ilegal"}), 400
        
    # Sanitasi nama file untuk mencegah directory traversal attacks
    _ = secure_filename(file.filename)
    
    # Validasi isi payload bytes gambar asli (Anti-Malware upload protection)
    if file.content_type.startswith('image/') and not is_safe_image_content(file.stream):
        return jsonify({"error": "Eksekusi ditolak: Konten file gambar tidak aman"}), 403
        
    try:
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal mendekode gambar"}), 400
        
        # TAHAP 1: Deteksi Mobil Utama (classes=[0]) dengan resolusi tinggi 1024
        results_cars = model(img, device=device, half=use_half, iou=0.4, conf=0.40, imgsz=1024, classes=[0], verbose=False)
        cars, plates = process_cascade(img, results_cars, is_tracking=False)
        
        return jsonify({
            "cars": cars, 
            "plates": plates, 
            "stats": {
                "total_vehicles": len(cars), 
                "with_plate": sum(1 for c in cars if c['plate_detected']), 
                "without_plate": len(cars) - sum(1 for c in cars if c['plate_detected'])
            }
        })
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/api/detect-frame', methods=['POST'])
@require_api_key
@limiter.limit("180 per minute")
def detect_frame():
    if 'file' not in request.files: 
        return jsonify({"error": "Data kosong"}), 400
        
    file = request.files['file']
    _ = secure_filename(file.filename)
    
    try:
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal mendekode frame video"}), 400
        
        # TAHAP 1: Tracking Multi-Object khusus Mobil (classes=[0]) via ByteTrack
        results_cars = model.track(
            img, 
            persist=True, 
            device=device, 
            half=use_half, 
            iou=0.4, 
            conf=0.40, 
            imgsz=1024, 
            classes=[0], 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        cars, plates = process_cascade(img, results_cars, is_tracking=True)
        
        return jsonify({
            "cars": cars, 
            "plates": plates,
            "stats": {
                "total_vehicles": len(cars), 
                "with_plate": sum(1 for c in cars if c['plate_detected']), 
                "without_plate": len(cars) - sum(1 for c in cars if c['plate_detected'])
            }
        })
    except Exception as e:
        return jsonify({"error": f"Gagal memproses frame: {str(e)}"}), 500

if __name__ == '__main__':
    # Mode WSGI Gunicorn akan meng-override baris ini saat berjalan di cluster produksi Render
    app.run(debug=False, host='0.0.0.0', port=5000)