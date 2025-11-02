"""
3D Print Model Detection and Masking System
Detects and masks ONLY the 3D printed model on the print bed
"""

import cv2
import numpy as np
from collections import deque
import time
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

class ModelDetector:
    def __init__(self):
        """Initialize 3D model detector"""
        
        # Background subtraction for model detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,        # Longer history for stable background
            varThreshold=30,    # Optimized for model detection
            detectShadows=False # Disable shadows for cleaner masks
        )
        
        # Model tracking
        self.model_history = deque(maxlen=50)
        self.baseline_established = False
        self.baseline_frames = 30
        
        # Model properties
        self.current_model_mask = None
        self.current_model_contour = None
        self.model_area = 0
        self.model_center = (0, 0)
        self.model_bounds = None
        
        # Processing parameters
        self.min_model_area = 1000      # Minimum area for model detection
        self.max_model_area = 100000    # Maximum area for model detection
        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        logger.info("3D Model Detector initialized")
    
    def detect_model(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Detect and mask the 3D printed model in the frame
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple of (model_mask, detection_info)
        """
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.noise_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.noise_kernel)
        
        # Find contours (potential model parts)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (remove noise, keep model)
        model_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_model_area < area < self.max_model_area:
                model_contours.append(contour)
        
        # Create model mask
        model_mask = np.zeros_like(fg_mask)
        
        if model_contours:
            # Combine all model contours
            cv2.fillPoly(model_mask, model_contours, 255)
            
            # Get largest contour as main model
            largest_contour = max(model_contours, key=cv2.contourArea)
            self.current_model_contour = largest_contour
            
            # Calculate model properties
            self.model_area = cv2.contourArea(largest_contour)
            
            # Model center
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                self.model_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # Model bounding box
            self.model_bounds = cv2.boundingRect(largest_contour)
            
        else:
            self.current_model_contour = None
            self.model_area = 0
            self.model_center = (0, 0)
            self.model_bounds = None
        
        # Store current mask
        self.current_model_mask = model_mask
        
        # Update history
        self.model_history.append(self.model_area)
        
        # Detection info
        detection_info = {
            'model_detected': len(model_contours) > 0,
            'model_area': self.model_area,
            'model_center': self.model_center,
            'model_bounds': self.model_bounds,
            'contour_count': len(model_contours),
            'total_contours': len(contours)
        }
        
        return model_mask, detection_info
    
    def get_model_only_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Get frame showing ONLY the 3D model (background removed)
        
        Args:
            frame: Original camera frame
            
        Returns:
            Frame with only the model visible
        """
        if self.current_model_mask is None:
            return np.zeros_like(frame)
        
        # Apply mask to show only model
        model_only = cv2.bitwise_and(frame, frame, mask=self.current_model_mask)
        
        return model_only
    
    def get_model_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Get frame with model highlighted/outlined
        
        Args:
            frame: Original camera frame
            
        Returns:
            Frame with model overlay
        """
        overlay_frame = frame.copy()
        
        if self.current_model_contour is not None:
            # Draw model outline
            cv2.drawContours(overlay_frame, [self.current_model_contour], -1, (0, 255, 0), 3)
            
            # Draw model center
            cv2.circle(overlay_frame, self.model_center, 5, (0, 255, 0), -1)
            
            # Draw bounding box
            if self.model_bounds:
                x, y, w, h = self.model_bounds
                cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add model info text
            cv2.putText(overlay_frame, f"Model Area: {self.model_area}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(overlay_frame, f"Center: {self.model_center}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay_frame
    
    def get_model_analysis(self) -> dict:
        """
        Get detailed model analysis
        
        Returns:
            Dictionary with model analysis data
        """
        
        analysis = {
            'model_detected': self.current_model_contour is not None,
            'model_area': self.model_area,
            'model_center': self.model_center,
            'model_bounds': self.model_bounds,
            'area_history': list(self.model_history),
            'area_stability': 0.0,
            'growth_rate': 0.0
        }
        
        # Calculate area stability (lower variance = more stable)
        if len(self.model_history) > 10:
            recent_areas = list(self.model_history)[-10:]
            if len(recent_areas) > 1:
                area_variance = np.var(recent_areas)
                area_mean = np.mean(recent_areas)
                if area_mean > 0:
                    analysis['area_stability'] = 1.0 - min(1.0, area_variance / area_mean)
        
        # Calculate growth rate (area change over time)
        if len(self.model_history) > 20:
            old_area = np.mean(list(self.model_history)[-20:-10])
            new_area = np.mean(list(self.model_history)[-10:])
            if old_area > 0:
                analysis['growth_rate'] = (new_area - old_area) / old_area
        
        return analysis
    
    def is_printing_active(self) -> bool:
        """
        Determine if printing is currently active based on model changes
        
        Returns:
            True if printing is detected
        """
        if len(self.model_history) < 10:
            return False
        
        # Check for consistent model presence
        recent_areas = list(self.model_history)[-10:]
        non_zero_areas = [area for area in recent_areas if area > self.min_model_area]
        
        # Printing is active if model is consistently present and growing
        model_present = len(non_zero_areas) > 7  # 70% of recent frames
        
        if model_present and len(self.model_history) > 20:
            # Check for growth (area increase over time)
            old_avg = np.mean(list(self.model_history)[-20:-10])
            new_avg = np.mean(list(self.model_history)[-10:])
            is_growing = new_avg > old_avg * 1.02  # 2% growth threshold
            
            return is_growing
        
        return model_present
    
    def reset_baseline(self):
        """Reset detector baseline"""
        self.model_history.clear()
        self.baseline_established = False
        self.current_model_mask = None
        self.current_model_contour = None
        logger.info("Model detector baseline reset")

class ModelMaskingSystem:
    """Complete system for 3D model detection and masking"""
    
    def __init__(self, camera_url: str):
        self.camera_url = camera_url
        self.cap = cv2.VideoCapture(camera_url)
        self.model_detector = ModelDetector()
        
        # Display settings
        self.display_mode = "overlay"  # "original", "mask", "model_only", "overlay"
        
        logger.info(f"Model Masking System initialized with camera: {camera_url}")
    
    def process_frame(self) -> Tuple[bool, np.ndarray, dict]:
        """
        Process single frame for model detection
        
        Returns:
            Tuple of (success, processed_frame, analysis)
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, {}
        
        # Detect model
        model_mask, detection_info = self.model_detector.detect_model(frame)
        
        # Get analysis
        analysis = self.model_detector.get_model_analysis()
        analysis.update(detection_info)
        
        # Generate display frame based on mode
        if self.display_mode == "original":
            display_frame = frame
        elif self.display_mode == "mask":
            display_frame = cv2.cvtColor(model_mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == "model_only":
            display_frame = self.model_detector.get_model_only_frame(frame)
        else:  # overlay
            display_frame = self.model_detector.get_model_overlay(frame)
        
        return True, display_frame, analysis
    
    def run_detection_loop(self):
        """Run real-time model detection loop"""
        logger.info("Starting model detection loop...")
        
        try:
            while True:
                ret, display_frame, analysis = self.process_frame()
                
                if ret:
                    # Show frame
                    cv2.imshow('3D Model Detection System', display_frame)
                    
                    # Print analysis (optional)
                    if analysis.get('model_detected'):
                        print(f"Model Area: {analysis['model_area']}, "
                              f"Center: {analysis['model_center']}, "
                              f"Printing: {self.model_detector.is_printing_active()}")
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('1'):
                        self.display_mode = "original"
                        print("Display mode: Original")
                    elif key == ord('2'):
                        self.display_mode = "mask"
                        print("Display mode: Mask")
                    elif key == ord('3'):
                        self.display_mode = "model_only"
                        print("Display mode: Model Only")
                    elif key == ord('4'):
                        self.display_mode = "overlay"
                        print("Display mode: Overlay")
                    elif key == ord('r'):
                        self.model_detector.reset_baseline()
                        print("Baseline reset")
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            logger.info("Detection loop interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Model detection system cleaned up")

if __name__ == "__main__":
    # Configuration
    CAMERA_URL = "http://192.168.1.17/webcam/?action=stream"
    
    # Create and run system
    system = ModelMaskingSystem(CAMERA_URL)
    
    print("="*60)
    print("3D Model Detection System")
    print("="*60)
    print("Controls:")
    print("  1 - Original frame")
    print("  2 - Model mask")
    print("  3 - Model only")
    print("  4 - Model overlay")
    print("  r - Reset baseline")
    print("  q - Quit")
    print("="*60)
    
    system.run_detection_loop()
