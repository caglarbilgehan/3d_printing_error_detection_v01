import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, Response, jsonify, request, session
from main import PrintStatusDetector
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from error_detection import EnhancedErrorDetectionSystem
from model_detector import ModelDetector
from translations import get_all_translations
import cv2
import numpy as np
import time
import threading
import requests
from functools import wraps

# OctoPrint Configuration (Raspberry Pi)
OCTOPRINT_API_KEY = "09C668315A784B138FF05305A5DF4E3F"
OCTOPRINT_URL = "http://192.168.1.17"  # Raspberry Pi OctoPrint adresi
OCTOPRINT_PORT = 80  # OctoPrint default port (genellikle 80 veya 5000)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
detector = PrintStatusDetector("http://192.168.1.17/webcam/?action=stream")
error_detector = EnhancedErrorDetectionSystem()
model_detector = ModelDetector()

# Global değişkenler
start_time = time.time()
frame_count = 0
current_frame = None
current_mask = None
current_graph = None
current_errors = {}
current_roi_mask = None
roi_points = []  # ROI noktaları
lock = threading.Lock()

# Performans ayarları
FRAME_SKIP = 2  # Her 2 frame'den birini işle
JPEG_QUALITY = 85  # JPEG kalitesi (1-100)
RESIZE_FACTOR = 1.0  # Görüntü boyutu (1.0 = orijinal, 0.5 = yarı)

current_status = {
    'is_printing': False,
    'motion_ratio': 0.0,
    'frame_count': 0,
    'uptime': 0
}

def process_frames():
    """Arka planda frame işleme - Optimize edilmiş"""
    global frame_count, current_status, current_frame, current_mask, current_graph, current_errors, current_roi_mask
    
    skip_counter = 0
    
    while True:
        ret, frame = detector.cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        
        # Frame atlama (performans için)
        skip_counter += 1
        if skip_counter % FRAME_SKIP != 0:
            continue
            
        frame_count += 1
        
        # Resize (performans için)
        if RESIZE_FACTOR < 1.0:
            frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_LINEAR)
        
        # ROI maskesi uygula
        processing_frame = frame.copy()
        roi_mask_visual = None
        if len(roi_points) >= 3:
            processing_frame, roi_mask_visual = detector.apply_roi_mask(processing_frame)
        
        # Hareket maskesi oluştur (sadece ROI içinde)
        fg_mask = detector.bg_subtractor.apply(processing_frame)
        
        # Motion ratio hesapla
        if roi_mask_visual is not None:
            # Sadece ROI içindeki hareketi say
            motion_pixels = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask_visual)
            motion_ratio = np.count_nonzero(motion_pixels) / np.count_nonzero(roi_mask_visual) if np.count_nonzero(roi_mask_visual) > 0 else 0
        else:
            motion_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        
        detector.motion_history.append(motion_ratio)
        printing = detector.is_printing()
        
        # Global durumu güncelle
        current_status['is_printing'] = printing
        current_status['motion_ratio'] = motion_ratio * 100
        current_status['frame_count'] = frame_count
        current_status['uptime'] = int(time.time() - start_time)
        
        # Hata tespiti yap (sadece printing sırasında)
        if printing and frame_count % 3 == 0:  # Her 3 frame'de bir
            # Model detection - 3D baskı modelini tespit et
            model_mask, model_info = model_detector.detect_model(frame)
            
            # Error detection (sadece model üzerinde)
            if model_mask is not None:
                current_errors = error_detector.analyze_frame(frame, model_mask)
            
            # Grafik oluştur
            current_graph = create_motion_graph()
        elif not printing:
            with lock:
                current_errors = {'total_errors': 0, 'errors': [], 'details': {}, 'baseline_established': False}
        
        # Grafik oluştur (her 5 frame'de bir)
        if frame_count % 5 == 0:
            graph = np.zeros((380, 1180, 3), dtype=np.uint8)
            graph[:] = (40, 40, 40)
            detector.draw_motion_graph(graph, 10, 10, 1160, 360)
        else:
            graph = current_graph if current_graph is not None else np.zeros((380, 1180, 3), dtype=np.uint8)
        
        with lock:
            current_frame = frame.copy()
            current_mask = fg_mask.copy()
            current_graph = graph
            current_roi_mask = roi_mask_visual

# Arka plan thread'i başlat
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

def generate_original_feed():
    """Orijinal kamera görüntüsü - ROI ile"""
    while True:
        with lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            frame = current_frame.copy()
            roi_mask = current_roi_mask
        
        # ROI çizimi
        if len(roi_points) >= 3:
            pts = np.array(roi_points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, "3D Print Area", (roi_points[0][0], roi_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # JPEG kalitesi ile encode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)  # 25 FPS

def generate_mask_feed():
    """Hareket maskesi görüntüsü - Optimize"""
    while True:
        with lock:
            if current_mask is None:
                time.sleep(0.05)
                continue
            mask = current_mask.copy()
        
        # Mask'ı renkli yap
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # JPEG kalitesi ile encode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', mask_colored, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

def generate_graph_feed():
    """Hareket grafiği görüntüsü - Optimize"""
    while True:
        with lock:
            if current_graph is None:
                time.sleep(0.05)
                continue
            graph = current_graph.copy()
        
        # JPEG kalitesi ile encode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', graph, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)  # Grafik daha yavaş güncellenebilir

@app.route('/')
def index():
    lang = session.get('language', 'tr')
    return render_template('dashboard.html', t=get_all_translations(lang), lang=lang)

@app.route('/octoprint')
def octoprint_page():
    lang = session.get('language', 'tr')
    return render_template('octoprint.html', t=get_all_translations(lang), lang=lang)

@app.route('/roi-setup')
def roi_setup_page():
    lang = session.get('language', 'tr')
    return render_template('roi_setup.html', t=get_all_translations(lang), lang=lang)

@app.route('/documentation')
def documentation_page():
    lang = session.get('language', 'tr')
    return render_template('documentation.html', t=get_all_translations(lang), lang=lang)

@app.route('/set-language/<lang>')
def set_language(lang):
    if lang in ['tr', 'en']:
        session['language'] = lang
    return jsonify({'success': True, 'language': lang})

# Hata Maskesi API Endpoint'leri
@app.route('/api/error-masks/<error_type>')
def get_error_mask(error_type):
    """Belirli bir hata tipinin maskesini döndür"""
    if error_type not in ['separation', 'underextrusion', 'deformation', 'surface_defect', 'model_deviation']:
        return jsonify({'error': 'Invalid error type'}), 400
    
    mask = error_detector.get_error_mask(error_type)
    if mask is not None:
        # Mask'ı JPEG olarak encode et
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', mask, encode_param)
        
        def generate():
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Boş siyah görüntü döndür
        empty = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', empty)
        
        def generate():
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ROI Overlay Feed
@app.route('/video_feed/model_mask')
def model_mask_feed():
    """3D model mask feed - sadece model gösterir"""
    def generate_model_mask():
        global current_frame
        while True:
            if current_frame is not None:
                frame = current_frame.copy()
                
                # Model maskesi oluştur
                model_mask, _ = model_detector.detect_model(frame)
                
                # Mask'ı renkli yap
                model_mask_colored = cv2.cvtColor(model_mask, cv2.COLOR_GRAY2BGR)
                
                # Frame'i resize et
                if RESIZE_FACTOR != 1.0:
                    height, width = model_mask_colored.shape[:2]
                    new_width = int(width * RESIZE_FACTOR)
                    new_height = int(height * RESIZE_FACTOR)
                    model_mask_colored = cv2.resize(model_mask_colored, (new_width, new_height))
                
                # JPEG kalitesi ile encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                ret, buffer = cv2.imencode('.jpg', model_mask_colored, encode_param)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)
    
    return Response(generate_model_mask(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/model_only')
def model_only_feed():
    """Sadece 3D model gösterir (arka plan kaldırılmış)"""
    def generate_model_only():
        global current_frame
        while True:
            if current_frame is not None:
                frame = current_frame.copy()
                
                # Sadece modeli göster
                model_only_frame = model_detector.get_model_only_frame(frame)
                
                # Frame'i resize et
                if RESIZE_FACTOR != 1.0:
                    height, width = model_only_frame.shape[:2]
                    new_width = int(width * RESIZE_FACTOR)
                    new_height = int(height * RESIZE_FACTOR)
                    model_only_frame = cv2.resize(model_only_frame, (new_width, new_height))
                
                # JPEG kalitesi ile encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                ret, buffer = cv2.imencode('.jpg', model_only_frame, encode_param)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)
    
    return Response(generate_model_only(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/model_overlay')
def model_overlay_feed():
    """Model overlay ile original feed"""
    def generate_model_overlay():
        global current_frame
        while True:
            if current_frame is not None:
                frame = current_frame.copy()
                
                # Model overlay ekle
                overlay_frame = model_detector.get_model_overlay(frame)
                
                # Frame'i resize et
                if RESIZE_FACTOR != 1.0:
                    height, width = overlay_frame.shape[:2]
                    new_width = int(width * RESIZE_FACTOR)
                    new_height = int(height * RESIZE_FACTOR)
                    overlay_frame = cv2.resize(overlay_frame, (new_width, new_height))
                
                # JPEG kalitesi ile encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                ret, buffer = cv2.imencode('.jpg', overlay_frame, encode_param)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)
    
    return Response(generate_model_overlay(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/original')
def original_feed():
    return Response(generate_original_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/mask')
def mask_feed():
    return Response(generate_mask_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/graph')
def graph_feed():
    return Response(generate_graph_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Gerçek zamanlı durum bilgisi API endpoint'i"""
    with lock:
        status_with_errors = current_status.copy()
        status_with_errors['errors'] = current_errors
    return jsonify(status_with_errors)

@app.route('/api/errors')
def get_errors():
    with lock:
        return jsonify(current_errors)

@app.route('/api/model')
def get_model_analysis():
    """3D model analizi"""
    analysis = model_detector.get_model_analysis()
    analysis['is_printing_active'] = model_detector.is_printing_active()
    return jsonify(analysis)

@app.route('/api/roi', methods=['GET', 'POST'])
def roi_management():
    """ROI (3D baskı alanı) yönetimi"""
    global roi_points
    
    if request.method == 'POST':
        data = request.json
        if 'points' in data:
            with lock:
                roi_points = data['points']
                detector.set_roi_mask(roi_points)
            return jsonify({'success': True, 'points': roi_points})
        elif 'reset' in data:
            with lock:
                roi_points = []
            return jsonify({'success': True, 'points': []})
    
    with lock:
        return jsonify({'points': roi_points})

@app.route('/api/performance', methods=['GET', 'POST'])
def performance_settings():
    """Performans ayarları"""
    global FRAME_SKIP, JPEG_QUALITY, RESIZE_FACTOR
    
    if request.method == 'POST':
        data = request.json
        if 'frame_skip' in data:
            FRAME_SKIP = max(1, min(5, data['frame_skip']))
        if 'jpeg_quality' in data:
            JPEG_QUALITY = max(50, min(100, data['jpeg_quality']))
        if 'resize_factor' in data:
            RESIZE_FACTOR = max(0.5, min(1.0, data['resize_factor']))
        
        return jsonify({
            'success': True,
            'frame_skip': FRAME_SKIP,
            'jpeg_quality': JPEG_QUALITY,
            'resize_factor': RESIZE_FACTOR
        })
    
    return jsonify({
        'frame_skip': FRAME_SKIP,
        'jpeg_quality': JPEG_QUALITY,
        'resize_factor': RESIZE_FACTOR
    })

# ==================== OctoPrint API Functions ====================

def octoprint_request(endpoint, method='GET', data=None):
    """OctoPrint API'ye istek gönder (Raspberry Pi'ye)"""
    try:
        headers = {'X-Api-Key': OCTOPRINT_API_KEY}
        url = f"{OCTOPRINT_URL}/api/{endpoint}"
        
        print(f"[OctoPrint] {method} {url}")  # Debug için
        
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data, timeout=10)
        
        print(f"[OctoPrint] Response: {response.status_code}")  # Debug için
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 204:  # No content (başarılı ama veri yok)
            return {'success': True}
        else:
            print(f"[OctoPrint] Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        print(f"[OctoPrint] Timeout connecting to {OCTOPRINT_URL}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[OctoPrint] Connection error to {OCTOPRINT_URL}")
        return None
    except Exception as e:
        print(f"[OctoPrint] API Error: {e}")
        return None

# 1. Printer Status & Connection
@app.route('/api/octoprint/connection')
def octoprint_connection():
    """Yazıcı bağlantı durumu"""
    return jsonify(octoprint_request('connection') or {})

@app.route('/api/octoprint/printer')
def octoprint_printer():
    """Yazıcı durumu (sıcaklık, pozisyon vb.)"""
    return jsonify(octoprint_request('printer') or {})

# 2. Job Information
@app.route('/api/octoprint/job')
def octoprint_job():
    """Aktif baskı işi bilgileri"""
    return jsonify(octoprint_request('job') or {})

# 3. Files
@app.route('/api/octoprint/files')
def octoprint_files():
    """Yüklü dosyalar listesi"""
    return jsonify(octoprint_request('files') or {})

# 4. System Commands
@app.route('/api/octoprint/system/commands')
def octoprint_system_commands():
    """Sistem komutları"""
    return jsonify(octoprint_request('system/commands') or {})

# 5. Printer Profiles
@app.route('/api/octoprint/printerprofiles')
def octoprint_profiles():
    """Yazıcı profilleri"""
    return jsonify(octoprint_request('printerprofiles') or {})

# 6. Settings
@app.route('/api/octoprint/settings')
def octoprint_settings():
    """OctoPrint ayarları"""
    return jsonify(octoprint_request('settings') or {})

# 7. Version Info
@app.route('/api/octoprint/version')
def octoprint_version():
    """OctoPrint versiyon bilgisi"""
    return jsonify(octoprint_request('version') or {})

# 8. Timelapse
@app.route('/api/octoprint/timelapse')
def octoprint_timelapse():
    """Timelapse ayarları ve dosyaları"""
    return jsonify(octoprint_request('timelapse') or {})

# 9. Plugin Manager
@app.route('/api/octoprint/plugin/pluginmanager')
def octoprint_plugins():
    """Yüklü plugin'ler"""
    return jsonify(octoprint_request('plugin/pluginmanager') or {})

# 10. Logs
@app.route('/api/octoprint/logs')
def octoprint_logs():
    """Log dosyaları"""
    return jsonify(octoprint_request('logs') or {})

# ==================== Control Commands ====================

@app.route('/api/octoprint/job/start', methods=['POST'])
def start_job():
    """Baskıyı başlat"""
    result = octoprint_request('job', method='POST', data={'command': 'start'})
    return jsonify({'success': result is not None})

@app.route('/api/octoprint/job/pause', methods=['POST'])
def pause_job():
    """Baskıyı duraklat"""
    result = octoprint_request('job', method='POST', data={'command': 'pause', 'action': 'pause'})
    return jsonify({'success': result is not None})

@app.route('/api/octoprint/job/resume', methods=['POST'])
def resume_job():
    """Baskıyı devam ettir"""
    result = octoprint_request('job', method='POST', data={'command': 'pause', 'action': 'resume'})
    return jsonify({'success': result is not None})

@app.route('/api/octoprint/job/cancel', methods=['POST'])
def cancel_job():
    """Baskıyı iptal et"""
    result = octoprint_request('job', method='POST', data={'command': 'cancel'})
    return jsonify({'success': result is not None})

@app.route('/api/octoprint/printer/command', methods=['POST'])
def send_gcode():
    """G-code komutu gönder"""
    command = request.json.get('command')
    if command:
        result = octoprint_request('printer/command', method='POST', data={'command': command})
        return jsonify({'success': result is not None})
    return jsonify({'success': False, 'error': 'No command provided'})

@app.route('/api/octoprint/printer/home', methods=['POST'])
def home_printer():
    """Yazıcıyı home pozisyonuna gönder"""
    axes = request.json.get('axes', ['x', 'y', 'z'])
    result = octoprint_request('printer/printhead', method='POST', data={'command': 'home', 'axes': axes})
    return jsonify({'success': result is not None})

@app.route('/api/octoprint/printer/temperature', methods=['POST'])
def set_temperature():
    """Sıcaklık ayarla"""
    tool = request.json.get('tool', 'tool0')
    target = request.json.get('target', 0)
    result = octoprint_request('printer/tool', method='POST', data={'command': 'target', 'targets': {tool: target}})
    return jsonify({'success': result is not None})

@app.route('/api/octoprint/printer/bed', methods=['POST'])
def set_bed_temperature():
    """Tabla sıcaklığı ayarla"""
    target = request.json.get('target', 0)
    result = octoprint_request('printer/bed', method='POST', data={'command': 'target', 'target': target})
    return jsonify({'success': result is not None})

# ==================== Advanced Features ====================

@app.route('/api/octoprint/dashboard')
def octoprint_dashboard():
    """Tüm önemli verileri tek seferde al"""
    return jsonify({
        'connection': octoprint_request('connection'),
        'printer': octoprint_request('printer'),
        'job': octoprint_request('job'),
        'version': octoprint_request('version'),
    })

if __name__ == '__main__':
    print("="*60)
    print("3D Printer Monitoring System")
    print("="*60)
    print(f"OctoPrint: {OCTOPRINT_URL}")
    print(f"Camera: http://192.168.1.17/webcam/?action=stream")
    print(f"Web Interface: http://localhost:5001")
    print(f"Running on PC, connecting to Raspberry Pi")
    print("="*60)
    
    # OctoPrint bağlantısını test et
    print("\nTesting OctoPrint connection...")
    test_result = octoprint_request('version')
    if test_result:
        print(f"OctoPrint connected! Version: {test_result.get('server', 'Unknown')}")
    else:
        print("OctoPrint connection failed! Check:")
        print(f"   - Raspberry Pi is online: {OCTOPRINT_URL}")
        print(f"   - API key is correct: {OCTOPRINT_API_KEY[:8]}...")
        print(f"   - Network connectivity")
    
    print("\nStarting Flask server on port 5001...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
