import sys
import os
import logging

# Reduce logging noise from detection modules
logging.getLogger('model_detector').setLevel(logging.WARNING)
logging.getLogger('main').setLevel(logging.WARNING)
logging.getLogger('error_detection').setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import psutil for system stats
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System stats will show simulated values.")

from flask import Flask, render_template, Response, jsonify, request, session, redirect
from main import PrintStatusDetector
from error_detection import EnhancedErrorDetectionSystem
from model_detector import ModelDetector
from octoprint_api import create_octoprint_api
from translations import get_all_translations
from config import load_config, save_config, get_octoprint_url, get_octoprint_api_key, get_camera_url, update_octoprint_settings
import cv2
import numpy as np
import time
import threading
import requests
from functools import wraps

# Load configuration
config = load_config()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Initialize with config
detector = PrintStatusDetector(get_camera_url())
error_detector = EnhancedErrorDetectionSystem()
model_detector = ModelDetector()
octoprint_api = create_octoprint_api(
    api_key=get_octoprint_api_key(),
    base_url=config['octoprint']['url'],
    port=config['octoprint']['port']
)


def reinitialize_connections():
    """Reinitialize detector and OctoPrint API with current config"""
    global detector, octoprint_api, config
    config = load_config()
    detector = PrintStatusDetector(get_camera_url())
    octoprint_api = create_octoprint_api(
        api_key=get_octoprint_api_key(),
        base_url=config['octoprint']['url'],
        port=config['octoprint']['port']
    )

# Global değişkenler
start_time = time.time()
frame_count = 0
current_frame = None
current_mask = None
current_graph = None
current_errors = {'total_errors': 0, 'detected_errors': [], 'details': {}, 'baseline_established': False}
current_roi_mask = None
roi_points = []  # ROI noktaları
lock = threading.Lock()

# Performans ayarları - Optimize edilmiş
FRAME_SKIP = 4  # Her 4 frame'den birini işle (daha az CPU kullanımı)
JPEG_QUALITY = 70  # JPEG kalitesi (düşürüldü)
RESIZE_FACTOR = 0.75  # Görüntü boyutu küçültüldü

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
        
        # Hata tespiti yap (sadece printing sırasında, daha az sıklıkta)
        if printing and frame_count % 10 == 0:  # Her 10 frame'de bir (CPU tasarrufu)
            # Model detection - 3D baskı modelini tespit et
            model_mask, model_info = model_detector.detect_model(frame)
            
            # Fallback: Eğer model bulunamazsa motion mask'ı kullan
            if model_mask is not None and np.count_nonzero(model_mask) == 0:
                # Motion mask'ı model mask olarak kullan
                model_mask = fg_mask.copy()
                logger.debug("Using motion mask as model mask fallback")
            
            # Error detection (sadece model üzerinde)
            if model_mask is not None:
                error_results = error_detector.analyze_frame(frame, model_mask)
                # Convert to JSON serializable format
                current_errors = {
                    'total_errors': error_results.get('total_errors', 0),
                    'detected_errors': error_results.get('detected_errors', []),
                    'baseline_established': error_results.get('baseline_established', False),
                    'details': {}
                }
                
                # Convert error details to serializable format
                if 'details' in error_results:
                    for error_type, details in error_results['details'].items():
                        if hasattr(details, '__dict__'):
                            # Convert dataclass to dict
                            current_errors['details'][error_type] = {
                                'detected': bool(getattr(details, 'detected', False)),
                                'confidence': float(getattr(details, 'confidence', 0.0)),
                                'severity': str(getattr(details, 'severity', 'low')),
                                'description': str(getattr(details, 'description', '')),
                                'timestamp': float(getattr(details, 'timestamp', 0.0))
                            }
                        else:
                            current_errors['details'][error_type] = details
            
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
        time.sleep(0.1)  # 10 FPS - CPU tasarrufu

def generate_mask_feed():
    """Hareket maskesi görüntüsü - Optimize"""
    while True:
        with lock:
            if current_mask is None:
                time.sleep(0.1)
                continue
            mask = current_mask.copy()
        
        # Mask'ı renkli yap
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # JPEG kalitesi ile encode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', mask_colored, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.15)  # ~7 FPS

def generate_graph_feed():
    """Hareket grafiği görüntüsü - Optimize"""
    while True:
        with lock:
            if current_graph is None:
                time.sleep(0.2)
                continue
            graph = current_graph.copy()
        
        # JPEG kalitesi ile encode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', graph, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.5)  # 2 FPS - Grafik için yeterli

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
    """3D model mask feed - cached mask kullanır (CPU tasarrufu)"""
    def generate_model_mask():
        global current_frame, current_mask
        while True:
            with lock:
                if current_mask is None:
                    time.sleep(0.2)
                    continue
                # Cached motion mask'ı kullan (model detection yapmadan)
                model_mask = current_mask.copy()
            
            # Mask'ı renkli yap
            model_mask_colored = cv2.cvtColor(model_mask, cv2.COLOR_GRAY2BGR)
            
            # JPEG kalitesi ile encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode('.jpg', model_mask_colored, encode_param)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.2)  # 5 FPS
    
    return Response(generate_model_mask(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/model_only')
def model_only_feed():
    """Sadece 3D model gösterir - cached mask kullanır (CPU tasarrufu)"""
    def generate_model_only():
        global current_frame, current_mask
        while True:
            with lock:
                if current_frame is None or current_mask is None:
                    time.sleep(0.2)
                    continue
                frame = current_frame.copy()
                model_mask = current_mask.copy()
            
            # Sadece modeli göster
            if np.count_nonzero(model_mask) > 0:
                model_only_frame = cv2.bitwise_and(frame, frame, mask=model_mask)
            else:
                model_only_frame = frame.copy()
            
            # JPEG kalitesi ile encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode('.jpg', model_only_frame, encode_param)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.2)  # 5 FPS
    
    return Response(generate_model_only(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/model_overlay')
def model_overlay_feed():
    """Model overlay - cached mask kullanır (CPU tasarrufu)"""
    def generate_model_overlay():
        global current_frame, current_mask
        while True:
            with lock:
                if current_frame is None or current_mask is None:
                    time.sleep(0.2)
                    continue
                frame = current_frame.copy()
                mask = current_mask.copy()
            
            overlay_frame = frame.copy()
            
            # Motion mask'tan contour çıkar ve overlay yap
            if np.count_nonzero(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 100:
                        cv2.drawContours(overlay_frame, [largest_contour], -1, (0, 255, 0), 2)
                        
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(overlay_frame, (cx, cy), 4, (0, 255, 0), -1)
                        
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode('.jpg', overlay_frame, encode_param)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.2)  # 5 FPS
    
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

def make_json_serializable(obj):
    """Convert numpy types and other non-serializable types to Python native types"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return obj

@app.route('/api/status')
def get_status():
    """Gerçek zamanlı durum bilgisi API endpoint'i"""
    with lock:
        status_with_errors = current_status.copy()
        # JSON serializable errors
        if isinstance(current_errors, dict):
            status_with_errors['errors'] = current_errors
        else:
            status_with_errors['errors'] = {'total_errors': 0, 'errors': [], 'details': {}}
    
    # Add system info
    if PSUTIL_AVAILABLE:
        try:
            status_with_errors['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
                'disk_free_gb': round((psutil.disk_usage('/').free if os.name != 'nt' else psutil.disk_usage('C:\\').free) / (1024**3), 1)
            }
        except Exception as e:
            status_with_errors['system'] = {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'disk_free_gb': 0
            }
    else:
        # Simulated values when psutil is not available
        import random
        status_with_errors['system'] = {
            'cpu_percent': round(random.uniform(20, 50), 1),
            'memory_percent': round(random.uniform(40, 70), 1),
            'disk_percent': round(random.uniform(30, 60), 1),
            'disk_free_gb': round(random.uniform(50, 200), 1)
        }
    
    return jsonify(make_json_serializable(status_with_errors))

@app.route('/api/errors')
def get_errors():
    with lock:
        # Ensure JSON serializable
        serializable_errors = {}
        if isinstance(current_errors, dict):
            for key, value in current_errors.items():
                if isinstance(value, bool):
                    serializable_errors[key] = bool(value)
                elif hasattr(value, '__dict__'):
                    # Convert dataclass to dict
                    serializable_errors[key] = {
                        'detected': bool(getattr(value, 'detected', False)),
                        'confidence': float(getattr(value, 'confidence', 0.0)),
                        'severity': str(getattr(value, 'severity', 'low')),
                        'description': str(getattr(value, 'description', '')),
                        'timestamp': float(getattr(value, 'timestamp', 0.0))
                    }
                else:
                    serializable_errors[key] = value
        else:
            serializable_errors = current_errors
        
        return jsonify(serializable_errors)

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
        current_config = load_config()
        api_key = current_config['octoprint']['api_key']
        base_url = get_octoprint_url()
        
        headers = {'X-Api-Key': api_key}
        url = f"{base_url}/api/{endpoint}"
        
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
        print(f"[OctoPrint] Timeout connecting to {base_url}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[OctoPrint] Connection error to {base_url}")
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
# ==================== COMPREHENSIVE OCTOPRINT API ====================

@app.route('/api/octoprint/comprehensive')
def get_comprehensive_octoprint_status():
    """Kapsamlı OctoPrint durumu"""
    try:
        status = octoprint_api.get_comprehensive_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting comprehensive status: {e}")
        return jsonify({'error': str(e)})

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

@app.route('/api/octoprint/gcode', methods=['POST'])
def send_gcode():
    """G-code komutları gönder"""
    data = request.json
    commands = data.get('commands', [])
    if isinstance(commands, str):
        commands = [commands]
    success = octoprint_api.send_gcode(commands)
    return jsonify({'success': success})

@app.route('/api/octoprint/connection', methods=['POST'])
def connection_control():
    """Printer bağlantı kontrolü"""
    data = request.json
    command = data.get('command')
    
    if command == 'connect':
        success = octoprint_api.connect_printer()
    elif command == 'disconnect':
        success = octoprint_api.disconnect_printer()
    else:
        return jsonify({'success': False, 'error': 'Invalid command'})
    
    return jsonify({'success': success})

@app.route('/api/octoprint/printer/sd', methods=['POST'])
def sd_control():
    """SD kart kontrolü"""
    data = request.json
    command = data.get('command')
    
    if command == 'init':
        success = octoprint_api.init_sd_card()
    elif command == 'refresh':
        success = octoprint_api.refresh_sd_files()
    elif command == 'release':
        success = octoprint_api.release_sd_card()
    else:
        return jsonify({'success': False, 'error': 'Invalid command'})
    
    return jsonify({'success': success})

@app.route('/api/octoprint/system/restart', methods=['POST'])
def restart_octoprint():
    """OctoPrint yeniden başlat"""
    success = octoprint_api.restart_octoprint()
    return jsonify({'success': success})

@app.route('/api/octoprint/system/shutdown', methods=['POST'])
def shutdown_system():
    """Sistemi kapat"""
    success = octoprint_api.shutdown_system()
    return jsonify({'success': success})

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


# ==================== LEGACY SETTINGS REDIRECT ====================

@app.route('/settings')
def settings_page():
    """Redirect old settings page to printers management"""
    return redirect('/printers')


# ==================== Printers Management API ====================

@app.route('/printers')
def printers_page():
    """Yazıcı yönetim sayfası"""
    lang = session.get('language', 'tr')
    translations = get_all_translations()
    t = translations.get(lang, translations['tr'])
    return render_template('printers.html', t=t, lang=lang)


@app.route('/api/printer-systems')
def api_printer_systems():
    """Get predefined printer system configurations"""
    from printers_config import PRINTER_SYSTEM_CONFIGS
    return jsonify(PRINTER_SYSTEM_CONFIGS)


@app.route('/api/printers', methods=['GET', 'POST'])
def api_printers():
    """Yazıcı listesi ve ekleme"""
    from printers_config import get_printers_manager, PrinterConfig
    
    manager = get_printers_manager()
    
    if request.method == 'GET':
        printers = []
        for printer in manager.get_all_printers():
            printers.append(printer.to_dict())
        return jsonify({'printers': printers})
    
    # POST - Add new printer
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    import uuid
    printer_id = f"printer_{uuid.uuid4().hex[:8]}"
    
    try:
        printer = PrinterConfig(
            id=printer_id,
            name=data.get('name', 'New Printer'),
            printer_type=data.get('printer_type', 'octoprint'),
            host=data.get('host', ''),
            port=int(data.get('port', 5000)),
            api_key=data.get('api_key', ''),
            brand=data.get('brand'),
            model=data.get('model'),
            serial_number=data.get('serial_number'),
            camera_url=data.get('camera_url'),
            ssl=data.get('ssl', 'false') == 'true',
            group_id=data.get('group_id')
        )
        
        if manager.add_printer(printer):
            return jsonify({'success': True, 'printer_id': printer_id})
        else:
            return jsonify({'success': False, 'error': 'Failed to add printer'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/printers/<printer_id>', methods=['GET', 'PUT', 'DELETE'])
def api_printer_detail(printer_id):
    """Tek yazıcı işlemleri"""
    from printers_config import get_printers_manager
    
    manager = get_printers_manager()
    
    if request.method == 'GET':
        printer = manager.get_printer(printer_id)
        if printer:
            return jsonify(printer.to_dict())
        return jsonify({'error': 'Printer not found'}), 404
    
    elif request.method == 'PUT':
        data = request.get_json()
        if manager.update_printer(printer_id, data):
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Failed to update'}), 500
    
    elif request.method == 'DELETE':
        if manager.remove_printer(printer_id):
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Failed to delete'}), 500


@app.route('/api/printers/test', methods=['POST'])
def api_printer_test():
    """Yazıcı bağlantı testi"""
    from printer_apis import get_printer_api
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    try:
        api = get_printer_api(
            printer_type=data.get('printer_type', 'octoprint'),
            host=data.get('host', ''),
            port=int(data.get('port', 5000)),
            api_key=data.get('api_key', ''),
            ssl=data.get('ssl', 'false') == 'true'
        )
        
        status = api.get_status()
        return jsonify({
            'success': status.state.value != 'offline',
            'state': status.state.value,
            'state_text': status.state_text
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/printers/<printer_id>/status')
def api_printer_status(printer_id):
    """Tek yazıcı durumu"""
    from printers_config import get_printers_manager
    from printer_apis import get_printer_api
    
    manager = get_printers_manager()
    printer = manager.get_printer(printer_id)
    
    if not printer:
        return jsonify({'error': 'Printer not found'}), 404
    
    try:
        api = get_printer_api(
            printer_type=printer.printer_type,
            host=printer.host,
            port=printer.port,
            api_key=printer.api_key,
            ssl=printer.ssl
        )
        
        status = api.get_status()
        return jsonify(status.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Cameras Management API ====================

@app.route('/api/cameras', methods=['GET', 'POST'])
def api_cameras():
    """Kamera listesi ve ekleme"""
    from printers_config import get_printers_manager, CameraConfig
    
    manager = get_printers_manager()
    
    if request.method == 'GET':
        cameras = []
        for camera in manager.get_all_cameras():
            cameras.append(camera.to_dict())
        return jsonify({'cameras': cameras})
    
    # POST - Add new camera
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    import uuid
    camera_id = f"camera_{uuid.uuid4().hex[:8]}"
    
    try:
        camera = CameraConfig(
            id=camera_id,
            name=data.get('name', 'New Camera'),
            url=data.get('url', ''),
            camera_type=data.get('camera_type', 'mjpeg'),
            printer_ids=data.get('printer_ids', []),
            username=data.get('username'),
            password=data.get('password'),
            position=data.get('position', 'front')
        )
        
        if manager.add_camera(camera):
            return jsonify({'success': True, 'camera_id': camera_id})
        else:
            return jsonify({'success': False, 'error': 'Failed to add camera'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def api_camera_delete(camera_id):
    """Kamera silme"""
    from printers_config import get_printers_manager
    
    manager = get_printers_manager()
    
    # Find and remove camera
    if camera_id in manager.cameras:
        del manager.cameras[camera_id]
        manager.save_config()
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Camera not found'}), 404


if __name__ == '__main__':
    from printers_config import get_printers_manager
    
    print("="*60)
    print("3D Printer Monitoring System - Multi-Printer Support")
    print("="*60)
    print(f"Web Interface: http://localhost:5001")
    print(f"Printer Management: http://localhost:5001/printers")
    print("="*60)
    
    # Load printers
    manager = get_printers_manager()
    printers = manager.get_all_printers()
    print(f"\nConfigured Printers: {len(printers)}")
    for p in printers:
        print(f"  - {p.name} ({p.printer_type}) @ {p.host}:{p.port}")
    
    # Test first printer connection if available
    if printers:
        print("\nTesting first printer connection...")
        octoprint_url = get_octoprint_url()
        test_result = octoprint_request('version')
        if test_result:
            print(f"OctoPrint connected! Version: {test_result.get('server', 'Unknown')}")
        else:
            print("Connection failed. Configure printers at /printers")
    else:
        print("\nNo printers configured. Add printers at /printers")
    
    print("\nStarting Flask server on port 5001...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
