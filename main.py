"""
Optimized 3D Printer Status Detection System
Enhanced for Raspberry Pi OctoPrint camera integration
"""

import cv2
import numpy as np
from collections import deque
import time
import logging
import threading
from typing import Optional, Tuple, List

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrintStatusDetector:
    def __init__(self, camera_url: str, roi_mask: Optional[np.ndarray] = None):
        """
        Initialize the 3D printer status detector
        
        Args:
            camera_url: URL of the camera stream (e.g., OctoPrint webcam)
            roi_mask: Optional region of interest mask
        """
        self.camera_url = camera_url
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Initialize camera with retry mechanism
        self._initialize_camera()
        
        # Background subtraction - Optimized for 3D printing
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,        # Longer history for stable background
            varThreshold=50,    # Higher threshold for noise reduction
            detectShadows=True  # Enable shadow detection for better accuracy
        )
        
        # Motion tracking
        self.motion_history = deque(maxlen=100)
        self.motion_threshold = 0.015  # Lower threshold for sensitive detection
        
        # ROI (Region of Interest) system
        self.roi_mask = roi_mask
        self.roi_points = []
        self.roi_area_percentage = 0.0
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Connection status
        self.is_connected = False
        self.last_frame_time = time.time()
        self.connection_timeout = 5.0  # seconds
        
        logger.info(f"PrintStatusDetector initialized with camera: {camera_url}")

    def _initialize_camera(self) -> bool:
        """Initialize camera connection with error handling"""
        try:
            # Release existing connection
            if self.cap is not None:
                self.cap.release()
            
            # Create new connection
            self.cap = cv2.VideoCapture(self.camera_url)
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
            self.cap.set(cv2.CAP_PROP_FPS, 25)        # Set target FPS
            
            # Test connection
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_height, self.frame_width = frame.shape[:2]
                self.is_connected = True
                logger.info(f"Camera connected successfully: {self.frame_width}x{self.frame_height}")
                return True
            else:
                logger.error("Failed to read test frame from camera")
                return False
                
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self.is_connected = False
            return False

    def reconnect_camera(self) -> bool:
        """Attempt to reconnect to camera"""
        logger.info("Attempting camera reconnection...")
        max_retries = 3
        
        for attempt in range(max_retries):
            if self._initialize_camera():
                logger.info(f"Camera reconnected successfully on attempt {attempt + 1}")
                return True
            
            time.sleep(2)  # Wait before retry
            
        logger.error("Camera reconnection failed after all attempts")
        return False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame with connection monitoring"""
        if not self.is_connected:
            if not self.reconnect_camera():
                return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                self.last_frame_time = time.time()
                return True, frame
            else:
                # Check if connection timed out
                if time.time() - self.last_frame_time > self.connection_timeout:
                    logger.warning("Camera connection timeout detected")
                    self.is_connected = False
                return False, None
                
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            self.is_connected = False
            return False, None

    def set_roi_points(self, points: List[List[int]]) -> bool:
        """
        Set ROI points and create mask
        
        Args:
            points: List of [x, y] coordinates defining the ROI polygon
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if len(points) < 3:
                self.roi_mask = None
                self.roi_points = []
                self.roi_area_percentage = 0.0
                logger.info("ROI mask cleared")
                return True
            
            try:
                # Create polygon mask
                mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
                # Calculate ROI area percentage
                roi_area = cv2.countNonZero(mask)
                total_area = self.frame_width * self.frame_height
                self.roi_area_percentage = (roi_area / total_area) * 100
                
                self.roi_mask = mask
                self.roi_points = points.copy()
                
                logger.info(f"ROI set with {len(points)} points, covering {self.roi_area_percentage:.1f}% of frame")
                return True
                
            except Exception as e:
                logger.error(f"Error setting ROI: {e}")
                return False

    def apply_roi_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply ROI mask to frame"""
        if self.roi_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
            return masked_frame, self.roi_mask
        return frame, None

    def detect_motion(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect motion in frame using background subtraction
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (motion_ratio, motion_mask)
        """
        start_time = time.time()
        
        # Apply ROI mask if available
        if self.roi_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion ratio
        if self.roi_mask is not None:
            # Only count motion within ROI
            roi_pixels = cv2.countNonZero(self.roi_mask)
            motion_pixels = cv2.countNonZero(cv2.bitwise_and(fg_mask, self.roi_mask))
            motion_ratio = motion_pixels / max(roi_pixels, 1)
        else:
            # Count motion in entire frame
            motion_ratio = cv2.countNonZero(fg_mask) / fg_mask.size
        
        # Store processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return motion_ratio, fg_mask

    def is_printing(self, min_frames: int = 10, threshold: float = None) -> bool:
        """
        Determine if printer is currently printing
        
        Args:
            min_frames: Minimum frames needed for decision
            threshold: Motion threshold (uses adaptive if None)
            
        Returns:
            bool: True if printing detected
        """
        if len(self.motion_history) < min_frames:
            return False
        
        # Use adaptive threshold based on ROI
        if threshold is None:
            if self.roi_area_percentage > 0:
                # Lower threshold for focused ROI
                threshold = 0.01
            else:
                # Higher threshold for full frame
                threshold = self.motion_threshold
        
        # Calculate recent motion average
        recent_motion = np.mean(list(self.motion_history)[-min_frames:])
        
        # Additional stability check - motion should be consistent
        motion_variance = np.var(list(self.motion_history)[-min_frames:])
        is_stable = motion_variance < 0.001  # Low variance indicates stable motion
        
        is_printing = recent_motion > threshold and is_stable
        
        if is_printing != getattr(self, '_last_printing_status', False):
            logger.info(f"Printing status changed: {is_printing} (motion: {recent_motion:.4f}, threshold: {threshold:.4f})")
            self._last_printing_status = is_printing
        
        return is_printing

    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        stats = {
            'fps': 0.0,
            'avg_processing_time': 0.0,
            'motion_history_size': len(self.motion_history),
            'roi_area_percentage': self.roi_area_percentage,
            'is_connected': self.is_connected,
            'frame_size': f"{self.frame_width}x{self.frame_height}"
        }
        
        if self.fps_counter:
            stats['fps'] = len(self.fps_counter) / max(sum(self.fps_counter), 0.001)
        
        if self.processing_times:
            stats['avg_processing_time'] = np.mean(self.processing_times) * 1000  # ms
        
        return stats

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame for motion detection and printing status
        
        Args:
            frame: Input frame from camera
            
        Returns:
            dict: Processing results
        """
        frame_start = time.time()
        
        # Detect motion
        motion_ratio, motion_mask = self.detect_motion(frame)
        
        # Update motion history
        with self.lock:
            self.motion_history.append(motion_ratio)
        
        # Determine printing status
        printing_status = self.is_printing()
        
        # Update FPS counter
        frame_time = time.time() - frame_start
        self.fps_counter.append(frame_time)
        
        return {
            'motion_ratio': motion_ratio,
            'motion_mask': motion_mask,
            'is_printing': printing_status,
            'processing_time': frame_time,
            'frame_timestamp': time.time()
        }

    def create_visualization(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Create visualization dashboard"""
        # Create dashboard layout
        dashboard_height = 800
        dashboard_width = 1200
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        dashboard[:] = (20, 20, 20)  # Dark background
        
        padding = 10
        
        # Original frame (top-left)
        frame_display = cv2.resize(frame, (580, 380))
        dashboard[padding:padding+380, padding:padding+580] = frame_display
        
        # Motion mask (top-right)
        motion_mask_colored = cv2.cvtColor(results['motion_mask'], cv2.COLOR_GRAY2BGR)
        mask_display = cv2.resize(motion_mask_colored, (580, 380))
        dashboard[padding:padding+380, 600+padding:600+padding+580] = mask_display
        
        # ROI overlay on original frame if available
        if self.roi_points:
            roi_overlay = frame_display.copy()
            pts = np.array([[int(p[0] * 580 / self.frame_width), 
                           int(p[1] * 380 / self.frame_height)] for p in self.roi_points], 
                          dtype=np.int32)
            cv2.polylines(roi_overlay, [pts], True, (0, 255, 0), 2)
            cv2.fillPoly(roi_overlay, [pts], (0, 255, 0), alpha=0.3)
            dashboard[padding:padding+380, padding:padding+580] = cv2.addWeighted(
                frame_display, 0.7, roi_overlay, 0.3, 0)
        
        # Motion graph (bottom)
        self._draw_motion_graph(dashboard, padding, 400, dashboard_width-2*padding, 380)
        
        # Status text
        status_text = "PRINTING" if results['is_printing'] else "IDLE"
        status_color = (0, 255, 0) if results['is_printing'] else (100, 100, 255)
        cv2.putText(dashboard, status_text, (padding, dashboard_height-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Performance stats
        stats = self.get_performance_stats()
        cv2.putText(dashboard, f"FPS: {stats['fps']:.1f}", (padding, dashboard_height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(dashboard, f"Motion: {results['motion_ratio']:.3f}", (200, dashboard_height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return dashboard

    def _draw_motion_graph(self, dashboard: np.ndarray, x: int, y: int, w: int, h: int):
        """Draw motion history graph"""
        # Background
        cv2.rectangle(dashboard, (x, y), (x+w, y+h), (40, 40, 40), -1)
        cv2.rectangle(dashboard, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        if len(self.motion_history) < 2:
            return
        
        # Grid lines
        for i in range(5):
            grid_y = y + int((i / 4) * h)
            cv2.line(dashboard, (x, grid_y), (x+w, grid_y), (60, 60, 60), 1)
        
        # Scale factor
        max_val = max(max(self.motion_history), 0.1)
        
        # Draw motion line
        points = []
        for i, motion in enumerate(self.motion_history):
            px = x + int((i / max(len(self.motion_history)-1, 1)) * w)
            py = y + h - int((motion / max_val) * h * 0.9)
            points.append((px, py))
        
        # Draw lines and points
        for i in range(1, len(points)):
            color = (0, 255, 255) if self.motion_history[i] > self.motion_threshold else (100, 100, 255)
            cv2.line(dashboard, points[i-1], points[i], color, 2)
        
        # Threshold line
        threshold_y = y + h - int((self.motion_threshold / max_val) * h * 0.9)
        cv2.line(dashboard, (x, threshold_y), (x+w, threshold_y), (255, 0, 0), 1)

    def run_detection_loop(self, display: bool = True):
        """Main detection loop"""
        logger.info("Starting detection loop...")
        
        try:
            while True:
                # Get frame
                ret, frame = self.get_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                results = self.process_frame(frame)
                
                # Display if requested
                if display:
                    dashboard = self.create_visualization(frame, results)
                    cv2.imshow('3D Printer Monitoring System', dashboard)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Detection loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    # Configuration
    CAMERA_URL = "http://192.168.1.17/webcam/?action=stream"
    
    # Create detector
    detector = PrintStatusDetector(CAMERA_URL)
    
    # Example ROI setup (optional)
    # roi_points = [[100, 100], [500, 100], [500, 400], [100, 400]]
    # detector.set_roi_points(roi_points)
    
    # Run detection
    detector.run_detection_loop(display=True)
