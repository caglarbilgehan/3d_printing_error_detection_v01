"""
Optimized Flask Web Application for 3D Printer Monitoring
Enhanced with threading, memory management, and performance optimizations
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, Response, jsonify, request, session
from main_optimized import PrintStatusDetector
from error_detection_enhanced import EnhancedErrorDetectionSystem
from translations import get_all_translations
import cv2
import numpy as np
import time
import threading
import requests
import logging
import gc
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Optional, Dict, Any
import psutil
import weakref

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # OctoPrint Configuration
    OCTOPRINT_API_KEY = "09C668315A784B138FF05305A5DF4E3F"
    OCTOPRINT_URL = "http://192.168.1.13"
    OCTOPRINT_PORT = 80
    CAMERA_URL = "http://192.168.1.13/webcam/?action=stream"
    
    # Performance Settings
    FRAME_SKIP = 2
    JPEG_QUALITY = 85
    RESIZE_FACTOR = 1.0
    MAX_FRAME_BUFFER_SIZE = 10
    PROCESSING_THREAD_COUNT = 2
    
    # Memory Management
    MEMORY_CLEANUP_INTERVAL = 300  # seconds
    MAX_MEMORY_USAGE_MB = 512
    
    # Flask Settings
    SECRET_KEY = 'your-secret-key-here-change-in-production'
    DEBUG = False

class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup"""
        gc.collect()
        logger.info(f"Memory cleanup completed. Current usage: {MemoryManager.get_memory_usage():.1f} MB")
    
    @staticmethod
    def check_memory_limit(limit_mb: float) -> bool:
        """Check if memory usage exceeds limit"""
        current_usage = MemoryManager.get_memory_usage()
        if current_usage > limit_mb:
            logger.warning(f"Memory usage ({current_usage:.1f} MB) exceeds limit ({limit_mb} MB)")
            return True
        return False

class FrameBuffer:
    """Thread-safe frame buffer with automatic cleanup"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.frames = {}
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def put(self, key: str, frame: np.ndarray):
        """Store frame with automatic size management"""
        with self.lock:
            # Clean old frames if buffer is full
            if len(self.frames) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                self.frames.pop(oldest_key, None)
                self.timestamps.pop(oldest_key, None)
            
            # Store new frame
            self.frames[key] = frame.copy() if frame is not None else None
            self.timestamps[key] = time.time()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get frame by key"""
        with self.lock:
            frame = self.frames.get(key)
            return frame.copy() if frame is not None else None
    
    def cleanup_old_frames(self, max_age: float = 30.0):
        """Remove frames older than max_age seconds"""
        with self.lock:
            current_time = time.time()
            old_keys = [k for k, t in self.timestamps.items() if current_time - t > max_age]
            
            for key in old_keys:
                self.frames.pop(key, None)
                self.timestamps.pop(key, None)
            
            if old_keys:
                logger.info(f"Cleaned up {len(old_keys)} old frames")

class OptimizedApp:
    """Optimized Flask application with enhanced performance"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = Config.SECRET_KEY
        
        # Initialize components
        self.detector = PrintStatusDetector(Config.CAMERA_URL)
        self.error_detector = EnhancedErrorDetectionSystem()
        
        # Thread-safe data structures
        self.frame_buffer = FrameBuffer(Config.MAX_FRAME_BUFFER_SIZE)
        self.status_lock = threading.RLock()
        self.roi_lock = threading.RLock()
        
        # Current state
        self.current_status = {
            'is_printing': False,
            'motion_ratio': 0.0,
            'frame_count': 0,
            'uptime': 0,
            'fps': 0.0,
            'memory_usage': 0.0
        }
        
        self.roi_points = []
        self.start_time = time.time()
        self.frame_count = 0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=Config.PROCESSING_THREAD_COUNT)
        self.processing_queue = queue.Queue(maxsize=50)
        self.shutdown_event = threading.Event()
        
        # Start background threads
        self._start_background_threads()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Optimized Flask application initialized")
    
    def _start_background_threads(self):
        """Start background processing threads"""
        # Frame processing thread
        self.processing_thread = threading.Thread(
            target=self._frame_processing_loop,
            daemon=True,
            name="FrameProcessor"
        )
        self.processing_thread.start()
        
        # Memory cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._memory_cleanup_loop,
            daemon=True,
            name="MemoryCleanup"
        )
        self.cleanup_thread.start()
        
        # Status update thread
        self.status_thread = threading.Thread(
            target=self._status_update_loop,
            daemon=True,
            name="StatusUpdater"
        )
        self.status_thread.start()
    
    def _frame_processing_loop(self):
        """Main frame processing loop"""
        logger.info("Frame processing loop started")
        frame_skip_counter = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Get frame from camera
                ret, frame = self.detector.get_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                # Frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter % Config.FRAME_SKIP != 0:
                    continue
                
                # Process frame
                self._process_single_frame(frame)
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in frame processing loop: {e}")
                time.sleep(1)
    
    def _process_single_frame(self, frame: np.ndarray):
        """Process a single frame"""
        try:
            # Update frame count
            self.frame_count += 1
            
            # Process with detector
            results = self.detector.process_frame(frame)
            
            # Error detection
            if results['motion_mask'] is not None:
                error_results = self.error_detector.analyze_frame(frame, results['motion_mask'])
            else:
                error_results = self.error_detector.get_error_summary()
            
            # Update status
            with self.status_lock:
                self.current_status.update({
                    'is_printing': results['is_printing'],
                    'motion_ratio': results['motion_ratio'],
                    'frame_count': self.frame_count,
                    'uptime': int(time.time() - self.start_time),
                    'fps': self.detector.get_performance_stats().get('fps', 0.0),
                    'memory_usage': MemoryManager.get_memory_usage()
                })
            
            # Store frames in buffer
            self.frame_buffer.put('original', frame)
            self.frame_buffer.put('motion_mask', results.get('motion_mask'))
            
            # Create visualization
            if results.get('motion_mask') is not None:
                graph_frame = self.detector.create_visualization(frame, results)
                self.frame_buffer.put('graph', graph_frame)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _memory_cleanup_loop(self):
        """Periodic memory cleanup"""
        logger.info("Memory cleanup loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Check memory usage
                if MemoryManager.check_memory_limit(Config.MAX_MEMORY_USAGE_MB):
                    MemoryManager.cleanup_memory()
                
                # Cleanup old frames
                self.frame_buffer.cleanup_old_frames()
                
                # Sleep until next cleanup
                time.sleep(Config.MEMORY_CLEANUP_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")
                time.sleep(60)
    
    def _status_update_loop(self):
        """Periodic status updates"""
        logger.info("Status update loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Update detector performance stats
                stats = self.detector.get_performance_stats()
                
                with self.status_lock:
                    self.current_status.update({
                        'fps': stats.get('fps', 0.0),
                        'memory_usage': MemoryManager.get_memory_usage(),
                        'roi_area_percentage': stats.get('roi_area_percentage', 0.0)
                    })
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in status update loop: {e}")
                time.sleep(5)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            lang = session.get('language', 'tr')
            return render_template('dashboard.html', t=get_all_translations(lang), lang=lang)
        
        @self.app.route('/octoprint')
        def octoprint_page():
            lang = session.get('language', 'tr')
            return render_template('octoprint.html', t=get_all_translations(lang), lang=lang)
        
        @self.app.route('/roi-setup')
        def roi_setup_page():
            lang = session.get('language', 'tr')
            return render_template('roi_setup.html', t=get_all_translations(lang), lang=lang)
        
        @self.app.route('/documentation')
        def documentation_page():
            lang = session.get('language', 'tr')
            return render_template('documentation.html', t=get_all_translations(lang), lang=lang)
        
        @self.app.route('/set-language/<lang>')
        def set_language(lang):
            if lang in ['tr', 'en']:
                session['language'] = lang
            return jsonify({'success': True, 'language': lang})
        
        # API Routes
        @self.app.route('/api/status')
        def get_status():
            with self.status_lock:
                return jsonify(self.current_status.copy())
        
        @self.app.route('/api/errors')
        def get_errors():
            return jsonify(self.error_detector.get_error_summary())
        
        @self.app.route('/api/performance')
        def get_performance():
            return jsonify({
                'frame_skip': Config.FRAME_SKIP,
                'jpeg_quality': Config.JPEG_QUALITY,
                'resize_factor': Config.RESIZE_FACTOR,
                'memory_usage': MemoryManager.get_memory_usage(),
                'detector_stats': self.detector.get_performance_stats()
            })
        
        @self.app.route('/api/performance', methods=['POST'])
        def update_performance():
            try:
                data = request.get_json()
                if 'frame_skip' in data:
                    Config.FRAME_SKIP = max(1, min(5, int(data['frame_skip'])))
                if 'jpeg_quality' in data:
                    Config.JPEG_QUALITY = max(50, min(100, int(data['jpeg_quality'])))
                if 'resize_factor' in data:
                    Config.RESIZE_FACTOR = max(0.5, min(1.0, float(data['resize_factor'])))
                
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/roi')
        def get_roi():
            with self.roi_lock:
                return jsonify({'points': self.roi_points})
        
        @self.app.route('/api/roi', methods=['POST'])
        def set_roi():
            try:
                data = request.get_json()
                
                if data.get('reset'):
                    with self.roi_lock:
                        self.roi_points = []
                    self.detector.set_roi_points([])
                    return jsonify({'success': True})
                
                points = data.get('points', [])
                if len(points) >= 3:
                    with self.roi_lock:
                        self.roi_points = points
                    self.detector.set_roi_points(points)
                    return jsonify({'success': True})
                else:
                    return jsonify({'success': False, 'error': 'Minimum 3 points required'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        # Video feed routes
        @self.app.route('/video_feed/original')
        def original_feed():
            return Response(self._generate_feed('original'), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/video_feed/mask')
        def mask_feed():
            return Response(self._generate_feed('motion_mask'), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/video_feed/graph')
        def graph_feed():
            return Response(self._generate_feed('graph'), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/video_feed/roi_overlay')
        def roi_overlay_feed():
            return Response(self._generate_roi_overlay_feed(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/error-masks/<error_type>')
        def get_error_mask(error_type):
            try:
                mask = self.error_detector.get_error_mask(error_type)
                if mask is not None:
                    return Response(self._generate_mask_feed(mask), 
                                  mimetype='multipart/x-mixed-replace; boundary=frame')
                else:
                    # Return empty black frame
                    empty = np.zeros((480, 640, 3), dtype=np.uint8)
                    return Response(self._generate_static_frame(empty), 
                                  mimetype='multipart/x-mixed-replace; boundary=frame')
            except Exception as e:
                logger.error(f"Error generating error mask for {error_type}: {e}")
                empty = np.zeros((480, 640, 3), dtype=np.uint8)
                return Response(self._generate_static_frame(empty), 
                              mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _generate_feed(self, feed_type: str):
        """Generate video feed"""
        while True:
            try:
                frame = self.frame_buffer.get(feed_type)
                if frame is not None:
                    # Resize if needed
                    if Config.RESIZE_FACTOR != 1.0:
                        height, width = frame.shape[:2]
                        new_width = int(width * Config.RESIZE_FACTOR)
                        new_height = int(height * Config.RESIZE_FACTOR)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert grayscale to BGR if needed
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    # Encode frame
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in feed generation ({feed_type}): {e}")
                time.sleep(1)
    
    def _generate_roi_overlay_feed(self):
        """Generate ROI overlay feed"""
        while True:
            try:
                frame = self.frame_buffer.get('original')
                if frame is not None and self.roi_points:
                    # Create ROI overlay
                    overlay_frame = self.detector.error_detector.get_roi_overlay(frame, self.roi_points)
                    
                    # Resize if needed
                    if Config.RESIZE_FACTOR != 1.0:
                        height, width = overlay_frame.shape[:2]
                        new_width = int(width * Config.RESIZE_FACTOR)
                        new_height = int(height * Config.RESIZE_FACTOR)
                        overlay_frame = cv2.resize(overlay_frame, (new_width, new_height))
                    
                    # Encode frame
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
                    ret, buffer = cv2.imencode('.jpg', overlay_frame, encode_param)
                    
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                elif frame is not None:
                    # No ROI, just return original
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if ret:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in ROI overlay feed: {e}")
                time.sleep(1)
    
    def _generate_mask_feed(self, mask: np.ndarray):
        """Generate feed for error mask"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', mask, encode_param)
        
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + b'\r\n')
    
    def _generate_static_frame(self, frame: np.ndarray):
        """Generate single static frame"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + b'\r\n')
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down application...")
        
        self.shutdown_event.set()
        
        # Wait for threads to finish
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5)
        if hasattr(self, 'cleanup_thread'):
            self.cleanup_thread.join(timeout=5)
        if hasattr(self, 'status_thread'):
            self.status_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Cleanup detector
        self.detector.cleanup()
        
        logger.info("Application shutdown completed")

# Create optimized app instance
optimized_app = OptimizedApp()
app = optimized_app.app

# OctoPrint integration (existing code)
def make_octoprint_request(endpoint, method='GET', data=None):
    """Make request to OctoPrint API with error handling"""
    try:
        headers = {
            'X-Api-Key': Config.OCTOPRINT_API_KEY,
            'Content-Type': 'application/json'
        }
        
        url = f"{Config.OCTOPRINT_URL}:{Config.OCTOPRINT_PORT}/api/{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=5)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data, timeout=5)
        else:
            return {'error': 'Unsupported method'}
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"OctoPrint API error: {response.status_code}")
            return {'error': f'API error: {response.status_code}'}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"OctoPrint connection error: {e}")
        return {'error': f'Connection error: {str(e)}'}

# Add OctoPrint routes
@app.route('/api/octoprint/<path:endpoint>')
def octoprint_proxy(endpoint):
    """Proxy requests to OctoPrint API"""
    result = make_octoprint_request(endpoint)
    return jsonify(result)

if __name__ == '__main__':
    try:
        logger.info("Starting optimized 3D Printer Monitoring System...")
        app.run(debug=Config.DEBUG, host='0.0.0.0', port=5001, threaded=True)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    finally:
        optimized_app.shutdown()
