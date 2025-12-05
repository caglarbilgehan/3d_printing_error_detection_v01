"""
Performance Monitoring and Configuration System
Real-time monitoring of system performance and automatic optimization
"""

import time
import psutil
import threading
import logging
import json
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    fps: float
    processing_time: float
    frame_drops: int
    error_detection_time: float
    network_latency: float

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.lock = threading.RLock()
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage_warning': 80.0,
            'cpu_usage_critical': 95.0,
            'memory_usage_warning': 85.0,
            'memory_usage_critical': 95.0,
            'fps_warning': 15.0,
            'fps_critical': 10.0,
            'processing_time_warning': 100.0,  # ms
            'processing_time_critical': 200.0,  # ms
        }
        
        # Auto-optimization settings
        self.auto_optimize = True
        self.optimization_cooldown = 30.0  # seconds
        self.last_optimization = 0.0
        
        # Current configuration
        self.current_config = {
            'frame_skip': 2,
            'jpeg_quality': 85,
            'resize_factor': 1.0,
            'roi_enabled': True,
            'error_detection_enabled': True
        }
        
        # Performance alerts
        self.alerts = deque(maxlen=100)
        
        logger.info("Performance monitor initialized")
    
    def record_metrics(self, 
                      fps: float = 0.0,
                      processing_time: float = 0.0,
                      frame_drops: int = 0,
                      error_detection_time: float = 0.0,
                      network_latency: float = 0.0):
        """Record performance metrics"""
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            fps=fps,
            processing_time=processing_time,
            frame_drops=frame_drops,
            error_detection_time=error_detection_time,
            network_latency=network_latency
        )
        
        # Store metrics
        with self.lock:
            self.metrics_history.append(metrics)
        
        # Check for performance issues
        self._check_performance_alerts(metrics)
        
        # Auto-optimize if needed
        if self.auto_optimize:
            self._auto_optimize(metrics)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        # CPU usage alerts
        if metrics.cpu_usage > self.thresholds['cpu_usage_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'cpu_usage',
                'value': metrics.cpu_usage,
                'threshold': self.thresholds['cpu_usage_critical'],
                'message': f"Critical CPU usage: {metrics.cpu_usage:.1f}%"
            })
        elif metrics.cpu_usage > self.thresholds['cpu_usage_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'cpu_usage',
                'value': metrics.cpu_usage,
                'threshold': self.thresholds['cpu_usage_warning'],
                'message': f"High CPU usage: {metrics.cpu_usage:.1f}%"
            })
        
        # Memory usage alerts
        if metrics.memory_usage > self.thresholds['memory_usage_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'memory_usage',
                'value': metrics.memory_usage,
                'threshold': self.thresholds['memory_usage_critical'],
                'message': f"Critical memory usage: {metrics.memory_usage:.1f}%"
            })
        elif metrics.memory_usage > self.thresholds['memory_usage_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'memory_usage',
                'value': metrics.memory_usage,
                'threshold': self.thresholds['memory_usage_warning'],
                'message': f"High memory usage: {metrics.memory_usage:.1f}%"
            })
        
        # FPS alerts
        if 0 < metrics.fps < self.thresholds['fps_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'fps',
                'value': metrics.fps,
                'threshold': self.thresholds['fps_critical'],
                'message': f"Critical low FPS: {metrics.fps:.1f}"
            })
        elif 0 < metrics.fps < self.thresholds['fps_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'fps',
                'value': metrics.fps,
                'threshold': self.thresholds['fps_warning'],
                'message': f"Low FPS: {metrics.fps:.1f}"
            })
        
        # Processing time alerts
        if metrics.processing_time > self.thresholds['processing_time_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'processing_time',
                'value': metrics.processing_time,
                'threshold': self.thresholds['processing_time_critical'],
                'message': f"Critical processing time: {metrics.processing_time:.1f}ms"
            })
        elif metrics.processing_time > self.thresholds['processing_time_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'processing_time',
                'value': metrics.processing_time,
                'threshold': self.thresholds['processing_time_warning'],
                'message': f"High processing time: {metrics.processing_time:.1f}ms"
            })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = metrics.timestamp
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert['message']}")
    
    def _auto_optimize(self, metrics: PerformanceMetrics):
        """Automatic performance optimization"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_optimization < self.optimization_cooldown:
            return
        
        optimization_applied = False
        
        # CPU optimization
        if metrics.cpu_usage > self.thresholds['cpu_usage_warning']:
            if self.current_config['frame_skip'] < 5:
                self.current_config['frame_skip'] += 1
                optimization_applied = True
                logger.info(f"Auto-optimization: Increased frame skip to {self.current_config['frame_skip']}")
            
            elif self.current_config['jpeg_quality'] > 60:
                self.current_config['jpeg_quality'] -= 10
                optimization_applied = True
                logger.info(f"Auto-optimization: Reduced JPEG quality to {self.current_config['jpeg_quality']}")
            
            elif self.current_config['resize_factor'] > 0.7:
                self.current_config['resize_factor'] -= 0.1
                optimization_applied = True
                logger.info(f"Auto-optimization: Reduced resolution to {self.current_config['resize_factor']:.1f}")
        
        # Memory optimization
        if metrics.memory_usage > self.thresholds['memory_usage_warning']:
            if self.current_config['error_detection_enabled']:
                # Temporarily disable error detection for memory relief
                self.current_config['error_detection_enabled'] = False
                optimization_applied = True
                logger.info("Auto-optimization: Temporarily disabled error detection")
        
        # FPS optimization
        if 0 < metrics.fps < self.thresholds['fps_warning']:
            if self.current_config['frame_skip'] < 4:
                self.current_config['frame_skip'] += 1
                optimization_applied = True
                logger.info(f"Auto-optimization: Increased frame skip to {self.current_config['frame_skip']} for FPS")
        
        if optimization_applied:
            self.last_optimization = current_time
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self, duration_seconds: int = 300) -> Dict:
        """Get performance metrics summary for specified duration"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        with self.lock:
            # Filter metrics within time range
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        fps_values = [m.fps for m in recent_metrics if m.fps > 0]
        processing_times = [m.processing_time for m in recent_metrics if m.processing_time > 0]
        
        summary = {
            'duration_seconds': duration_seconds,
            'sample_count': len(recent_metrics),
            'cpu_usage': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': np.mean(cpu_values) if cpu_values else 0,
                'max': np.max(cpu_values) if cpu_values else 0,
                'min': np.min(cpu_values) if cpu_values else 0
            },
            'memory_usage': {
                'current': memory_values[-1] if memory_values else 0,
                'average': np.mean(memory_values) if memory_values else 0,
                'max': np.max(memory_values) if memory_values else 0,
                'min': np.min(memory_values) if memory_values else 0
            },
            'fps': {
                'current': fps_values[-1] if fps_values else 0,
                'average': np.mean(fps_values) if fps_values else 0,
                'max': np.max(fps_values) if fps_values else 0,
                'min': np.min(fps_values) if fps_values else 0
            },
            'processing_time': {
                'current': processing_times[-1] if processing_times else 0,
                'average': np.mean(processing_times) if processing_times else 0,
                'max': np.max(processing_times) if processing_times else 0,
                'min': np.min(processing_times) if processing_times else 0
            }
        }
        
        return summary
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent performance alerts"""
        with self.lock:
            return list(self.alerts)[-count:]
    
    def update_configuration(self, config: Dict):
        """Update performance configuration"""
        self.current_config.update(config)
        logger.info(f"Configuration updated: {config}")
    
    def get_configuration(self) -> Dict:
        """Get current configuration"""
        return self.current_config.copy()
    
    def reset_optimization(self):
        """Reset to default configuration"""
        self.current_config = {
            'frame_skip': 2,
            'jpeg_quality': 85,
            'resize_factor': 1.0,
            'roi_enabled': True,
            'error_detection_enabled': True
        }
        self.last_optimization = 0.0
        logger.info("Configuration reset to defaults")
    
    def export_metrics(self, filename: str, duration_seconds: int = 3600):
        """Export metrics to JSON file"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        with self.lock:
            # Filter metrics within time range
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        # Convert to serializable format
        export_data = {
            'export_timestamp': current_time,
            'duration_seconds': duration_seconds,
            'metrics_count': len(recent_metrics),
            'metrics': [asdict(m) for m in recent_metrics],
            'alerts': list(self.alerts),
            'configuration': self.current_config,
            'thresholds': self.thresholds
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Metrics exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

class ConfigurationManager:
    """Configuration management for 3D printer monitoring"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'camera': {
                'url': 'http://192.168.1.13/webcam/?action=stream',
                'timeout': 5.0,
                'reconnect_attempts': 3,
                'buffer_size': 1
            },
            'processing': {
                'frame_skip': 2,
                'jpeg_quality': 85,
                'resize_factor': 1.0,
                'roi_enabled': True,
                'error_detection_enabled': True
            },
            'error_detection': {
                'baseline_frames': 50,
                'sensitivity': 'medium',  # low, medium, high
                'auto_reset_baseline': True,
                'baseline_reset_interval': 3600  # seconds
            },
            'performance': {
                'auto_optimize': True,
                'memory_limit_mb': 512,
                'cpu_limit_percent': 80,
                'cleanup_interval': 300
            },
            'octoprint': {
                'api_key': '09C668315A784B138FF05305A5DF4E3F',
                'url': 'http://192.168.1.13',
                'port': 80,
                'timeout': 5.0
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5001,
                'debug': False,
                'secret_key': 'your-secret-key-here-change-in-production'
            }
        }
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                self._merge_config(self.config, file_config)
            logger.info(f"Configuration loaded from {self.config_file}")
        except FileNotFoundError:
            logger.info("Configuration file not found, using defaults")
            self.save_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _merge_config(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated path"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """Set configuration value by dot-separated path"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key_path} = {value}")
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section"""
        return self.config.get(section, {}).copy()
    
    def update_section(self, section: str, updates: Dict):
        """Update configuration section"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
        logger.info(f"Configuration section '{section}' updated")

if __name__ == "__main__":
    # Test performance monitoring
    monitor = PerformanceMonitor()
    
    # Simulate some metrics
    for i in range(10):
        monitor.record_metrics(
            fps=25.0 - i,
            processing_time=50 + i * 10,
            frame_drops=i,
            error_detection_time=20 + i * 2
        )
        time.sleep(0.1)
    
    # Get summary
    summary = monitor.get_metrics_summary(60)
    print("Performance Summary:", summary)
    
    # Test configuration manager
    config_manager = ConfigurationManager()
    print("Camera URL:", config_manager.get('camera.url'))
    
    config_manager.set('processing.frame_skip', 3)
    config_manager.save_config()
