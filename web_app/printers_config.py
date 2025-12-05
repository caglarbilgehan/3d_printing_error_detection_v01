"""
Multi-Printer Configuration Module
Supports: OctoPrint, Repetier Server, Klipper/Moonraker, Duet, and custom implementations
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class PrinterType(Enum):
    OCTOPRINT = "octoprint"
    REPETIER = "repetier"
    KLIPPER = "klipper"  # Moonraker API
    DUET = "duet"
    CUSTOM = "custom"

# Predefined printer system configurations
PRINTER_SYSTEM_CONFIGS = {
    'octoprint': {
        'name': 'OctoPrint',
        'default_port': 5000,
        'api_key_required': True,
        'api_key_location': 'Settings > API > Global API Key',
        'camera_path': '/webcam/?action=stream',
        'description': 'OctoPrint is a web interface for 3D printers'
    },
    'repetier': {
        'name': 'Repetier Server',
        'default_port': 3344,
        'api_key_required': True,
        'api_key_location': 'Settings > Global Settings > API Key',
        'camera_path': '/printer/cammjpg/',
        'description': 'Repetier Server for multiple printer management'
    },
    'klipper': {
        'name': 'Klipper / Moonraker',
        'default_port': 7125,
        'api_key_required': False,
        'api_key_location': 'Not required by default',
        'camera_path': '/webcam/?action=stream',
        'description': 'Klipper firmware with Moonraker API'
    },
    'duet': {
        'name': 'Duet Web Control',
        'default_port': 80,
        'api_key_required': False,
        'api_key_location': 'Optional password in settings',
        'camera_path': '/webcam',
        'description': 'Duet boards with RepRapFirmware'
    },
    'custom': {
        'name': 'Custom / Other',
        'default_port': 80,
        'api_key_required': False,
        'api_key_location': 'Depends on system',
        'camera_path': '',
        'description': 'Custom printer system - configure manually'
    }
}

@dataclass
class PrinterConfig:
    """Configuration for a single printer"""
    id: str
    name: str
    printer_type: str  # PrinterType value
    host: str
    port: int
    api_key: str
    # Printer details
    brand: Optional[str] = None  # e.g., "Creality", "Prusa", "Voron"
    model: Optional[str] = None  # e.g., "Ender 3 Pro", "MK3S+"
    serial_number: Optional[str] = None
    # Camera
    camera_url: Optional[str] = None
    # Status
    enabled: bool = True
    # Connection settings
    timeout: int = 10
    ssl: bool = False
    # Group
    group_id: Optional[str] = None
    # Printer-specific settings
    extra_settings: Optional[Dict] = None
    
    def get_base_url(self) -> str:
        protocol = "https" if self.ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    def get_display_name(self) -> str:
        """Get display name with brand/model if available"""
        if self.brand and self.model:
            return f"{self.name} ({self.brand} {self.model})"
        return self.name
    
    def to_dict(self) -> Dict:
        return asdict(self)

class CameraType(Enum):
    MJPEG = "mjpeg"           # Standard MJPEG stream (OctoPrint, etc.)
    RTSP = "rtsp"             # RTSP stream (IP cameras)
    HLS = "hls"               # HLS stream
    SNAPSHOT = "snapshot"      # Snapshot URL (refresh based)
    TUYA = "tuya"             # Tuya smart camera
    GOOGLE_HOME = "google_home"  # Google Home camera
    ONVIF = "onvif"           # ONVIF compatible cameras
    USB = "usb"               # Local USB camera
    CUSTOM = "custom"         # Custom implementation

@dataclass
class CameraConfig:
    """Configuration for a camera (can be independent of printer)"""
    id: str
    name: str
    url: str
    camera_type: str = "mjpeg"  # CameraType value
    printer_ids: Optional[List[str]] = None  # Can monitor multiple printers
    enabled: bool = True
    # Stream settings
    username: Optional[str] = None
    password: Optional[str] = None
    refresh_rate: int = 1000  # For snapshot type, ms between refreshes
    # Position/Role
    position: str = "front"  # front, top, side, custom
    is_primary: bool = False  # Primary camera for the printer
    # Extra settings for specific camera types
    extra_settings: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        # Convert None lists to empty lists for JSON
        if data['printer_ids'] is None:
            data['printer_ids'] = []
        return data
    
    def get_stream_url(self) -> str:
        """Get the appropriate stream URL based on camera type"""
        if self.camera_type == CameraType.RTSP.value and self.username:
            # Insert credentials into RTSP URL
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.url)
            netloc = f"{self.username}:{self.password}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
        return self.url

class PrintersManager:
    """Manages multiple printer configurations"""
    
    CONFIG_FILE = "printers.json"
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(config_dir, self.CONFIG_FILE)
        self.printers: Dict[str, PrinterConfig] = {}
        self.cameras: Dict[str, CameraConfig] = {}
        self.load_config()
    
    def load_config(self):
        """Load printer configurations from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load printers
                for p_data in data.get('printers', []):
                    printer = PrinterConfig(**p_data)
                    self.printers[printer.id] = printer
                
                # Load cameras
                for c_data in data.get('cameras', []):
                    camera = CameraConfig(**c_data)
                    self.cameras[camera.id] = camera
                    
            except Exception as e:
                print(f"Error loading printers config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration with one OctoPrint printer"""
        # Try to load from existing settings.json for backward compatibility
        settings_path = os.path.join(os.path.dirname(self.config_path), 'settings.json')
        
        default_printer = PrinterConfig(
            id="printer_1",
            name="3D Yazıcı 1",
            printer_type=PrinterType.OCTOPRINT.value,
            host="192.168.1.100",
            port=5000,
            api_key="",
            camera_url="http://192.168.1.100/webcam/?action=stream"
        )
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                octoprint = settings.get('octoprint', {})
                default_printer.host = octoprint.get('url', '192.168.1.100')
                default_printer.port = octoprint.get('port', 5000)
                default_printer.api_key = octoprint.get('api_key', '')
                default_printer.camera_url = octoprint.get('camera_url', '')
            except:
                pass
        
        self.printers[default_printer.id] = default_printer
        
        # Add default camera
        default_camera = CameraConfig(
            id="camera_1",
            name="Ana Kamera",
            url=default_printer.camera_url or "",
            printer_ids=[default_printer.id]
        )
        self.cameras[default_camera.id] = default_camera
        
        self.save_config()
    
    def save_config(self):
        """Save printer configurations to file"""
        data = {
            'printers': [p.to_dict() for p in self.printers.values()],
            'cameras': [c.to_dict() for c in self.cameras.values()]
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving printers config: {e}")
    
    def add_printer(self, printer: PrinterConfig) -> bool:
        """Add a new printer"""
        if printer.id in self.printers:
            return False
        self.printers[printer.id] = printer
        self.save_config()
        return True
    
    def update_printer(self, printer_id: str, updates: Dict) -> bool:
        """Update an existing printer"""
        if printer_id not in self.printers:
            return False
        
        printer = self.printers[printer_id]
        for key, value in updates.items():
            if hasattr(printer, key):
                setattr(printer, key, value)
        
        self.save_config()
        return True
    
    def remove_printer(self, printer_id: str) -> bool:
        """Remove a printer"""
        if printer_id not in self.printers:
            return False
        del self.printers[printer_id]
        self.save_config()
        return True
    
    def get_printer(self, printer_id: str) -> Optional[PrinterConfig]:
        """Get a specific printer"""
        return self.printers.get(printer_id)
    
    def get_all_printers(self) -> List[PrinterConfig]:
        """Get all printers"""
        return list(self.printers.values())
    
    def get_enabled_printers(self) -> List[PrinterConfig]:
        """Get only enabled printers"""
        return [p for p in self.printers.values() if p.enabled]
    
    def add_camera(self, camera: CameraConfig) -> bool:
        """Add a new camera"""
        if camera.id in self.cameras:
            return False
        self.cameras[camera.id] = camera
        self.save_config()
        return True
    
    def get_all_cameras(self) -> List[CameraConfig]:
        """Get all cameras"""
        return list(self.cameras.values())
    
    def get_printer_cameras(self, printer_id: str) -> List[CameraConfig]:
        """Get cameras associated with a printer"""
        return [c for c in self.cameras.values() if c.printer_id == printer_id]


# Global instance
_printers_manager: Optional[PrintersManager] = None

def get_printers_manager() -> PrintersManager:
    """Get the global printers manager instance"""
    global _printers_manager
    if _printers_manager is None:
        _printers_manager = PrintersManager()
    return _printers_manager
