"""
Enhanced 3D Printer Error Detection System
Advanced computer vision algorithms for real-time print quality monitoring
"""

import cv2
import numpy as np
from collections import deque
import time
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """3D Printing error types"""
    SEPARATION = "separation"           # Warping/lifting from bed
    UNDEREXTRUSION = "underextrusion"   # Insufficient material flow
    DEFORMATION = "deformation"         # Shape distortion
    SURFACE_DEFECT = "surface_defect"   # Surface quality issues
    MODEL_DEVIATION = "model_deviation" # Size/structure deviation

@dataclass
class ErrorResult:
    """Error detection result"""
    detected: bool
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: float

class EnhancedErrorDetectionSystem:
    def __init__(self, baseline_frames: int = 50):
        """
        Initialize enhanced error detection system
        
        Args:
            baseline_frames: Number of frames to establish baseline
        """
        self.baseline_frames = baseline_frames
        self.frame_count = 0
        self.baseline_established = False
        
        # Enhanced baseline metrics
        self.baseline_metrics = {
            'contour_area': 0.0,
            'edge_density': 0.0,
            'motion_ratio': 0.0,
            'brightness_mean': 0.0,
            'brightness_std': 0.0,
            'texture_energy': 0.0
        }
        
        # History buffers with adaptive sizes
        self.contour_history = deque(maxlen=100)
        self.edge_history = deque(maxlen=100)
        self.motion_history = deque(maxlen=100)
        self.brightness_history = deque(maxlen=100)
        self.texture_history = deque(maxlen=100)
        
        # Error states with enhanced information
        self.errors = {}
        for error_type in ErrorType:
            self.errors[error_type.value] = ErrorResult(
                detected=False,
                confidence=0.0,
                severity='low',
                description='',
                timestamp=0.0
            )
        
        # Error masks for visualization
        self.error_masks = {error_type.value: None for error_type in ErrorType}
        
        # Enhanced detection parameters
        self.detection_params = {
            'separation': {
                'motion_variance_threshold': 0.0005,
                'contour_drop_threshold': 0.15,
                'min_confidence': 0.3
            },
            'underextrusion': {
                'motion_threshold': 0.03,
                'edge_ratio_threshold': 0.6,
                'consistency_frames': 15
            },
            'deformation': {
                'variation_coefficient_threshold': 0.25,
                'stability_frames': 20,
                'shape_deviation_threshold': 0.2
            },
            'surface_defect': {
                'texture_variance_threshold': 0.0002,
                'edge_variance_threshold': 0.0001,
                'brightness_variance_threshold': 50.0
            },
            'model_deviation': {
                'size_deviation_threshold': 0.12,
                'consistency_frames': 25,
                'trend_analysis_frames': 40
            }
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=50)
        self.last_frame = None
        self.last_motion_mask = None
        
        logger.info("Enhanced Error Detection System initialized")

    def extract_advanced_features(self, frame: np.ndarray, motion_mask: np.ndarray) -> Dict:
        """Extract advanced features from frame"""
        start_time = time.time()
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Edge detection with multiple methods
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        # Contour analysis
        contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
        total_contour_area = sum(cv2.contourArea(c) for c in significant_contours)
        
        # Shape analysis
        shape_features = self._analyze_shapes(significant_contours)
        
        # Texture analysis using Local Binary Patterns
        texture_features = self._analyze_texture(gray)
        
        # Brightness and contrast analysis
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Edge density
        edge_density = np.count_nonzero(edges_canny) / edges_canny.size
        
        # Motion analysis
        motion_ratio = np.count_nonzero(motion_mask) / motion_mask.size
        
        # Advanced motion features
        motion_features = self._analyze_motion_patterns(motion_mask)
        
        features = {
            'contour_area': total_contour_area,
            'edge_density': edge_density,
            'motion_ratio': motion_ratio,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'texture_energy': texture_features['energy'],
            'texture_contrast': texture_features['contrast'],
            'shape_complexity': shape_features['complexity'],
            'shape_regularity': shape_features['regularity'],
            'motion_coherence': motion_features['coherence'],
            'motion_direction': motion_features['direction'],
            'edges_canny': edges_canny,
            'edges_sobel': edges_sobel
        }
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return features

    def _analyze_shapes(self, contours: List) -> Dict:
        """Analyze shape characteristics"""
        if not contours:
            return {'complexity': 0.0, 'regularity': 1.0}
        
        complexities = []
        regularities = []
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Shape complexity (perimeter to area ratio)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 0:
                complexity = (perimeter ** 2) / (4 * np.pi * area)
                complexities.append(complexity)
            
            # Shape regularity (how close to perfect shape)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                regularity = area / hull_area
                regularities.append(regularity)
        
        return {
            'complexity': np.mean(complexities) if complexities else 0.0,
            'regularity': np.mean(regularities) if regularities else 1.0
        }

    def _analyze_texture(self, gray: np.ndarray) -> Dict:
        """Analyze texture using statistical methods"""
        # Calculate GLCM-like features
        # Simplified version for performance
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture energy and contrast
        energy = np.sum(gradient_magnitude**2) / gradient_magnitude.size
        contrast = np.std(gradient_magnitude)
        
        return {
            'energy': energy,
            'contrast': contrast
        }

    def _analyze_motion_patterns(self, motion_mask: np.ndarray) -> Dict:
        """Analyze motion patterns for coherence and direction"""
        # Find motion contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'coherence': 0.0, 'direction': 0.0}
        
        # Calculate motion coherence (how clustered the motion is)
        total_area = sum(cv2.contourArea(c) for c in contours)
        largest_contour_area = max(cv2.contourArea(c) for c in contours)
        
        coherence = largest_contour_area / max(total_area, 1)
        
        # Calculate dominant motion direction
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                direction = ellipse[2]  # Angle
            else:
                direction = 0.0
        else:
            direction = 0.0
        
        return {
            'coherence': coherence,
            'direction': direction
        }

    def analyze_frame(self, frame: np.ndarray, motion_mask: np.ndarray) -> Dict:
        """
        Analyze frame for error detection
        
        Args:
            frame: Input frame
            motion_mask: Motion detection mask
            
        Returns:
            Dict: Error detection results
        """
        self.frame_count += 1
        self.last_frame = frame.copy()
        self.last_motion_mask = motion_mask.copy()
        
        # Extract features
        features = self.extract_advanced_features(frame, motion_mask)
        
        # Update history
        self.contour_history.append(features['contour_area'])
        self.edge_history.append(features['edge_density'])
        self.motion_history.append(features['motion_ratio'])
        self.brightness_history.append(features['brightness_mean'])
        self.texture_history.append(features['texture_energy'])
        
        # Establish baseline
        if not self.baseline_established and len(self.contour_history) >= self.baseline_frames:
            self._establish_baseline()
        
        # Detect errors if baseline is ready
        if self.baseline_established:
            self._detect_all_errors(features)
        
        return self.get_error_summary()

    def _establish_baseline(self):
        """Establish baseline metrics from initial frames"""
        self.baseline_metrics = {
            'contour_area': np.mean(list(self.contour_history)),
            'edge_density': np.mean(list(self.edge_history)),
            'motion_ratio': np.mean(list(self.motion_history)),
            'brightness_mean': np.mean(list(self.brightness_history)),
            'brightness_std': np.std(list(self.brightness_history)),
            'texture_energy': np.mean(list(self.texture_history))
        }
        
        self.baseline_established = True
        logger.info(f"Baseline established after {self.baseline_frames} frames")
        logger.info(f"Baseline metrics: {self.baseline_metrics}")

    def _detect_all_errors(self, features: Dict):
        """Detect all error types"""
        self._detect_separation_enhanced(features)
        self._detect_underextrusion_enhanced(features)
        self._detect_deformation_enhanced(features)
        self._detect_surface_defects_enhanced(features)
        self._detect_model_deviation_enhanced(features)

    def _detect_separation_enhanced(self, features: Dict):
        """Enhanced separation/warping detection"""
        error_type = 'separation'
        params = self.detection_params[error_type]
        
        if len(self.motion_history) < 20:
            return
        
        # Motion variance analysis
        recent_motion = list(self.motion_history)[-20:]
        motion_variance = np.var(recent_motion)
        
        # Contour area drop analysis
        recent_contours = list(self.contour_history)[-10:]
        if self.baseline_metrics['contour_area'] > 0:
            contour_drop = (self.baseline_metrics['contour_area'] - np.mean(recent_contours)) / self.baseline_metrics['contour_area']
        else:
            contour_drop = 0
        
        # Shape regularity analysis
        shape_irregularity = 1.0 - features.get('shape_regularity', 1.0)
        
        # Combined detection logic
        motion_score = min(motion_variance / params['motion_variance_threshold'], 1.0)
        contour_score = max(0, contour_drop / params['contour_drop_threshold'])
        shape_score = shape_irregularity * 2.0
        
        # Weighted confidence
        confidence = (motion_score * 0.4 + contour_score * 0.4 + shape_score * 0.2)
        confidence = min(confidence, 0.95)
        
        # Detection threshold
        detected = confidence > params['min_confidence']
        
        # Determine severity
        if confidence > 0.8:
            severity = 'critical'
        elif confidence > 0.6:
            severity = 'high'
        elif confidence > 0.4:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Update error state
        self.errors[error_type] = ErrorResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            description=f"Warping detected: motion_var={motion_variance:.6f}, contour_drop={contour_drop:.3f}",
            timestamp=time.time()
        )
        
        # Create visualization mask
        if detected:
            self.error_masks[error_type] = cv2.bitwise_or(
                self.last_motion_mask,
                features['edges_canny']
            )
        else:
            self.error_masks[error_type] = np.zeros_like(self.last_motion_mask)

    def _detect_underextrusion_enhanced(self, features: Dict):
        """Enhanced under-extrusion detection"""
        error_type = 'underextrusion'
        params = self.detection_params[error_type]
        
        if len(self.motion_history) < params['consistency_frames']:
            return
        
        # Motion vs edge density analysis
        recent_motion = np.mean(list(self.motion_history)[-params['consistency_frames']:])
        recent_edges = np.mean(list(self.edge_history)[-params['consistency_frames']:])
        
        # Check for motion without material deposition
        has_motion = recent_motion > params['motion_threshold']
        
        if has_motion and self.baseline_metrics['edge_density'] > 0:
            edge_ratio = recent_edges / self.baseline_metrics['edge_density']
            
            # Confidence based on motion-edge mismatch
            if edge_ratio < params['edge_ratio_threshold']:
                confidence = min(0.95, (1.0 - edge_ratio) * (recent_motion / params['motion_threshold']))
                detected = True
                severity = 'critical' if confidence > 0.7 else 'high'
                description = f"Under-extrusion: motion={recent_motion:.3f}, edge_ratio={edge_ratio:.3f}"
            else:
                confidence = 0.0
                detected = False
                severity = 'low'
                description = "Normal extrusion detected"
        else:
            confidence = 0.0
            detected = False
            severity = 'low'
            description = "No motion or insufficient baseline"
        
        # Update error state
        self.errors[error_type] = ErrorResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            description=description,
            timestamp=time.time()
        )
        
        # Create visualization mask
        if detected:
            # Show areas with motion but no edges
            motion_without_edges = cv2.bitwise_and(
                self.last_motion_mask,
                cv2.bitwise_not(features['edges_canny'])
            )
            self.error_masks[error_type] = motion_without_edges
        else:
            self.error_masks[error_type] = np.zeros_like(self.last_motion_mask)

    def _detect_deformation_enhanced(self, features: Dict):
        """Enhanced deformation detection"""
        error_type = 'deformation'
        params = self.detection_params[error_type]
        
        if len(self.contour_history) < params['stability_frames']:
            return
        
        # Shape analysis
        recent_contours = list(self.contour_history)[-params['stability_frames']:]
        contour_variance = np.var(recent_contours)
        contour_mean = np.mean(recent_contours)
        
        # Shape complexity analysis
        shape_complexity = features.get('shape_complexity', 1.0)
        shape_regularity = features.get('shape_regularity', 1.0)
        
        # Variation coefficient
        if contour_mean > 0:
            variation_coefficient = np.sqrt(contour_variance) / contour_mean
        else:
            variation_coefficient = 0
        
        # Shape deviation score
        shape_deviation = (shape_complexity - 1.0) + (1.0 - shape_regularity)
        
        # Combined confidence
        variation_score = min(variation_coefficient / params['variation_coefficient_threshold'], 1.0)
        shape_score = min(shape_deviation / params['shape_deviation_threshold'], 1.0)
        
        confidence = (variation_score * 0.6 + shape_score * 0.4)
        confidence = min(confidence, 0.95)
        
        detected = confidence > 0.3
        
        # Severity assessment
        if confidence > 0.7:
            severity = 'high'
        elif confidence > 0.5:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Update error state
        self.errors[error_type] = ErrorResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            description=f"Deformation: var_coeff={variation_coefficient:.3f}, shape_dev={shape_deviation:.3f}",
            timestamp=time.time()
        )
        
        # Create visualization mask
        if detected:
            # Highlight edges with dilation to show deformed areas
            self.error_masks[error_type] = cv2.dilate(
                features['edges_canny'],
                np.ones((5, 5), np.uint8)
            )
        else:
            self.error_masks[error_type] = np.zeros_like(features['edges_canny'])

    def _detect_surface_defects_enhanced(self, features: Dict):
        """Enhanced surface defect detection"""
        error_type = 'surface_defect'
        params = self.detection_params[error_type]
        
        if len(self.texture_history) < 20:
            return
        
        # Texture variance analysis
        recent_texture = list(self.texture_history)[-20:]
        texture_variance = np.var(recent_texture)
        
        # Edge variance analysis
        recent_edges = list(self.edge_history)[-20:]
        edge_variance = np.var(recent_edges)
        
        # Brightness variance analysis
        recent_brightness = list(self.brightness_history)[-20:]
        brightness_variance = np.var(recent_brightness)
        
        # Multi-factor confidence
        texture_score = min(texture_variance / params['texture_variance_threshold'], 1.0)
        edge_score = min(edge_variance / params['edge_variance_threshold'], 1.0)
        brightness_score = min(brightness_variance / params['brightness_variance_threshold'], 1.0)
        
        confidence = (texture_score * 0.4 + edge_score * 0.4 + brightness_score * 0.2)
        confidence = min(confidence, 0.95)
        
        detected = confidence > 0.25
        
        # Severity based on multiple factors
        if confidence > 0.6:
            severity = 'medium'
        elif confidence > 0.4:
            severity = 'low'
        else:
            severity = 'low'
        
        # Update error state
        self.errors[error_type] = ErrorResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            description=f"Surface defects: texture_var={texture_variance:.6f}, edge_var={edge_variance:.6f}",
            timestamp=time.time()
        )
        
        # Create visualization mask
        if detected:
            # Use morphological operations to highlight surface irregularities
            self.error_masks[error_type] = cv2.morphologyEx(
                features['edges_canny'],
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8)
            )
        else:
            self.error_masks[error_type] = np.zeros_like(features['edges_canny'])

    def _detect_model_deviation_enhanced(self, features: Dict):
        """Enhanced model deviation detection"""
        error_type = 'model_deviation'
        params = self.detection_params[error_type]
        
        if len(self.contour_history) < params['trend_analysis_frames']:
            return
        
        # Trend analysis
        recent_contours = list(self.contour_history)[-params['trend_analysis_frames']:]
        baseline_contour = self.baseline_metrics['contour_area']
        
        if baseline_contour > 0:
            # Calculate deviation trend
            deviations = [(c - baseline_contour) / baseline_contour for c in recent_contours]
            mean_deviation = np.mean(np.abs(deviations))
            deviation_trend = np.polyfit(range(len(deviations)), deviations, 1)[0]  # Slope
            
            # Size consistency analysis
            size_consistency = 1.0 - (np.std(recent_contours) / max(np.mean(recent_contours), 1))
            
            # Combined confidence
            deviation_score = min(mean_deviation / params['size_deviation_threshold'], 1.0)
            trend_score = min(abs(deviation_trend) * 10, 1.0)  # Amplify trend effect
            consistency_score = 1.0 - max(0, size_consistency)
            
            confidence = (deviation_score * 0.5 + trend_score * 0.3 + consistency_score * 0.2)
            confidence = min(confidence, 0.95)
            
            detected = confidence > 0.2
            
            # Severity assessment
            if confidence > 0.6:
                severity = 'high'
            elif confidence > 0.4:
                severity = 'medium'
            else:
                severity = 'low'
            
            description = f"Model deviation: mean_dev={mean_deviation:.3f}, trend={deviation_trend:.6f}"
        else:
            confidence = 0.0
            detected = False
            severity = 'low'
            description = "Insufficient baseline data"
        
        # Update error state
        self.errors[error_type] = ErrorResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            description=description,
            timestamp=time.time()
        )
        
        # Create visualization mask
        if detected:
            self.error_masks[error_type] = features['edges_canny'].copy()
        else:
            self.error_masks[error_type] = np.zeros_like(features['edges_canny'])

    def get_error_summary(self) -> Dict:
        """Get comprehensive error summary"""
        detected_errors = [k for k, v in self.errors.items() if v.detected]
        
        # Calculate overall risk score
        risk_scores = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }
        
        total_risk = sum(risk_scores.get(error.severity, 0) * error.confidence 
                        for error in self.errors.values() if error.detected)
        
        return {
            'total_errors': len(detected_errors),
            'detected_errors': detected_errors,
            'details': {k: {
                'detected': v.detected,
                'confidence': v.confidence,
                'severity': v.severity,
                'description': v.description,
                'timestamp': v.timestamp
            } for k, v in self.errors.items()},
            'baseline_established': self.baseline_established,
            'overall_risk_score': min(total_risk, 1.0),
            'processing_stats': {
                'avg_processing_time': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
                'frames_processed': self.frame_count
            }
        }

    def get_error_mask(self, error_type: str) -> Optional[np.ndarray]:
        """Get visualization mask for specific error type"""
        if error_type in self.error_masks and self.error_masks[error_type] is not None:
            # Convert to colored mask
            mask = self.error_masks[error_type]
            if len(mask.shape) == 2:  # Grayscale mask
                colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # Color code by error type
                if error_type == 'separation':
                    colored_mask[:, :, 1] = 0  # Remove green
                    colored_mask[:, :, 0] = 0  # Remove blue (keep red)
                elif error_type == 'underextrusion':
                    colored_mask[:, :, 0] = 0  # Remove blue
                    colored_mask[:, :, 2] = 0  # Remove red (keep green)
                elif error_type == 'deformation':
                    colored_mask[:, :, 1] = 0  # Remove green
                    colored_mask[:, :, 2] = 0  # Remove red (keep blue)
                elif error_type == 'surface_defect':
                    colored_mask[:, :, 0] = colored_mask[:, :, 2]  # Yellow (red + green)
                elif error_type == 'model_deviation':
                    colored_mask[:, :, 1] = colored_mask[:, :, 2]  # Magenta (red + blue)
                
                return colored_mask
            return mask
        return None

    def reset_baseline(self):
        """Reset baseline for recalibration"""
        self.baseline_established = False
        self.frame_count = 0
        self.contour_history.clear()
        self.edge_history.clear()
        self.motion_history.clear()
        self.brightness_history.clear()
        self.texture_history.clear()
        
        # Reset error states
        for error_type in ErrorType:
            self.errors[error_type.value] = ErrorResult(
                detected=False,
                confidence=0.0,
                severity='low',
                description='',
                timestamp=0.0
            )
        
        logger.info("Baseline reset - recalibration started")

    def get_roi_overlay(self, frame: np.ndarray, roi_points: List[List[int]]) -> np.ndarray:
        """Create ROI overlay on frame"""
        if len(roi_points) < 3:
            return frame
        
        overlay = frame.copy()
        pts = np.array(roi_points, dtype=np.int32)
        
        # Yeşil şeffaf polygon
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
        
        # Blend
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # "3D Print Area" etiketi
        cv2.putText(result, "3D Print Area", (roi_points[0][0], roi_points[0][1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result

if __name__ == "__main__":
    # Test the enhanced error detection system
    detector = EnhancedErrorDetectionSystem()
    
    # Simulate some test data
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
    
    # Process test frame
    results = detector.analyze_frame(test_frame, test_mask)
    print("Test results:", results)
