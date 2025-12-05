"""
Comprehensive OctoPrint API Integration
Full API coverage for 3D printer monitoring and control
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PrinterState(Enum):
    """Printer state enumeration"""
    OPERATIONAL = "Operational"
    PRINTING = "Printing"
    PAUSED = "Paused"
    ERROR = "Error"
    OFFLINE = "Offline"
    CANCELLING = "Cancelling"
    PAUSING = "Pausing"
    RESUMING = "Resuming"

@dataclass
class OctoPrintConfig:
    """OctoPrint configuration"""
    api_key: str
    base_url: str
    port: int = 80
    timeout: int = 10

class ComprehensiveOctoPrintAPI:
    """Comprehensive OctoPrint API client"""
    
    def __init__(self, config: OctoPrintConfig):
        self.config = config
        self.base_url = f"{config.base_url}:{config.port}"
        self.headers = {
            'X-Api-Key': config.api_key,
            'Content-Type': 'application/json'
        }
        
        logger.info(f"OctoPrint API initialized: {self.base_url}")
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Optional[Dict]:
        """Make HTTP request to OctoPrint API"""
        url = f"{self.base_url}/api/{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, timeout=self.config.timeout)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=self.config.timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=self.config.timeout)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            if response.status_code == 200:
                return response.json() if response.content else {}
            elif response.status_code == 204:
                return {}  # No content success
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    # ==================== VERSION INFO ====================
    
    def get_version(self) -> Optional[Dict]:
        """Get OctoPrint version information"""
        return self._make_request('version')
    
    # ==================== CONNECTION ====================
    
    def get_connection_info(self) -> Optional[Dict]:
        """Get current connection settings"""
        return self._make_request('connection')
    
    def connect_printer(self, port: str = None, baudrate: int = None, profile: str = None) -> bool:
        """Connect to printer"""
        data = {'command': 'connect'}
        if port:
            data['port'] = port
        if baudrate:
            data['baudrate'] = baudrate
        if profile:
            data['printerProfile'] = profile
            
        result = self._make_request('connection', 'POST', data)
        return result is not None
    
    def disconnect_printer(self) -> bool:
        """Disconnect from printer"""
        data = {'command': 'disconnect'}
        result = self._make_request('connection', 'POST', data)
        return result is not None
    
    # ==================== PRINTER STATUS ====================
    
    def get_printer_status(self) -> Optional[Dict]:
        """Get current printer status"""
        return self._make_request('printer')
    
    def get_printer_profiles(self) -> Optional[Dict]:
        """Get printer profiles"""
        return self._make_request('printerprofiles')
    
    # ==================== JOB INFORMATION ====================
    
    def get_job_info(self) -> Optional[Dict]:
        """Get current job information"""
        return self._make_request('job')
    
    def start_job(self) -> bool:
        """Start current job"""
        data = {'command': 'start'}
        result = self._make_request('job', 'POST', data)
        return result is not None
    
    def pause_job(self, action: str = 'pause') -> bool:
        """Pause/Resume job (action: 'pause', 'resume', 'toggle')"""
        data = {'command': 'pause', 'action': action}
        result = self._make_request('job', 'POST', data)
        return result is not None
    
    def cancel_job(self) -> bool:
        """Cancel current job"""
        data = {'command': 'cancel'}
        result = self._make_request('job', 'POST', data)
        return result is not None
    
    def restart_job(self) -> bool:
        """Restart current job"""
        data = {'command': 'restart'}
        result = self._make_request('job', 'POST', data)
        return result is not None
    
    # ==================== FILES ====================
    
    def get_files(self, location: str = 'local', recursive: bool = True) -> Optional[Dict]:
        """Get files list"""
        params = f"?recursive={'true' if recursive else 'false'}"
        return self._make_request(f'files/{location}{params}')
    
    def get_file_info(self, location: str, filename: str) -> Optional[Dict]:
        """Get specific file information"""
        return self._make_request(f'files/{location}/{filename}')
    
    def select_file(self, location: str, filename: str, print_after_select: bool = False) -> bool:
        """Select file for printing"""
        data = {'command': 'select', 'print': print_after_select}
        result = self._make_request(f'files/{location}/{filename}', 'POST', data)
        return result is not None
    
    def delete_file(self, location: str, filename: str) -> bool:
        """Delete file"""
        result = self._make_request(f'files/{location}/{filename}', 'DELETE')
        return result is not None
    
    # ==================== SETTINGS ====================
    
    def get_settings(self) -> Optional[Dict]:
        """Get OctoPrint settings"""
        return self._make_request('settings')
    
    def update_settings(self, settings: Dict) -> bool:
        """Update OctoPrint settings"""
        result = self._make_request('settings', 'POST', settings)
        return result is not None
    
    # ==================== SYSTEM ====================
    
    def get_system_info(self) -> Optional[Dict]:
        """Get system information"""
        return self._make_request('system')
    
    def execute_system_command(self, command: str) -> bool:
        """Execute system command"""
        data = {'command': command}
        result = self._make_request('system/commands', 'POST', data)
        return result is not None
    
    def restart_octoprint(self) -> bool:
        """Restart OctoPrint"""
        return self.execute_system_command('restart')
    
    def shutdown_system(self) -> bool:
        """Shutdown system"""
        return self.execute_system_command('shutdown')
    
    def reboot_system(self) -> bool:
        """Reboot system"""
        return self.execute_system_command('reboot')
    
    # ==================== TEMPERATURE ====================
    
    def get_temperature_data(self) -> Optional[Dict]:
        """Get temperature data"""
        return self._make_request('printer/tool')
    
    def set_tool_temperature(self, tool: int, target: float) -> bool:
        """Set tool temperature"""
        data = {'command': 'target', 'targets': {f'tool{tool}': target}}
        result = self._make_request('printer/tool', 'POST', data)
        return result is not None
    
    def set_bed_temperature(self, target: float) -> bool:
        """Set bed temperature"""
        data = {'command': 'target', 'targets': {'bed': target}}
        result = self._make_request('printer/bed', 'POST', data)
        return result is not None
    
    # ==================== PRINT HEAD ====================
    
    def jog_printhead(self, x: float = None, y: float = None, z: float = None, speed: int = None) -> bool:
        """Jog print head"""
        data = {'command': 'jog'}
        if x is not None:
            data['x'] = x
        if y is not None:
            data['y'] = y
        if z is not None:
            data['z'] = z
        if speed is not None:
            data['speed'] = speed
            
        result = self._make_request('printer/printhead', 'POST', data)
        return result is not None
    
    def home_printhead(self, axes: List[str] = None) -> bool:
        """Home print head axes"""
        data = {'command': 'home'}
        if axes:
            data['axes'] = axes
        else:
            data['axes'] = ['x', 'y', 'z']
            
        result = self._make_request('printer/printhead', 'POST', data)
        return result is not None
    
    # ==================== SD CARD ====================
    
    def get_sd_state(self) -> Optional[Dict]:
        """Get SD card state"""
        return self._make_request('printer/sd')
    
    def init_sd_card(self) -> bool:
        """Initialize SD card"""
        data = {'command': 'init'}
        result = self._make_request('printer/sd', 'POST', data)
        return result is not None
    
    def refresh_sd_files(self) -> bool:
        """Refresh SD card files"""
        data = {'command': 'refresh'}
        result = self._make_request('printer/sd', 'POST', data)
        return result is not None
    
    def release_sd_card(self) -> bool:
        """Release SD card"""
        data = {'command': 'release'}
        result = self._make_request('printer/sd', 'POST', data)
        return result is not None
    
    # ==================== GCODE COMMANDS ====================
    
    def send_gcode(self, commands: List[str]) -> bool:
        """Send G-code commands"""
        data = {'commands': commands}
        result = self._make_request('printer/command', 'POST', data)
        return result is not None
    
    # ==================== PLUGINS ====================
    
    def get_plugins(self) -> Optional[Dict]:
        """Get installed plugins"""
        return self._make_request('plugins')
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict]:
        """Get specific plugin information"""
        return self._make_request(f'plugins/{plugin_id}')
    
    # ==================== USERS ====================
    
    def get_users(self) -> Optional[Dict]:
        """Get users list"""
        return self._make_request('users')
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user information"""
        return self._make_request('currentuser')
    
    # ==================== COMPREHENSIVE STATUS ====================
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive printer status"""
        status = {
            'timestamp': time.time(),
            'connection': self.get_connection_info(),
            'printer': self.get_printer_status(),
            'job': self.get_job_info(),
            'version': self.get_version(),
            'system': self.get_system_info(),
            'settings': None,  # Skip settings for performance
            'files': None,     # Skip files for performance
            'plugins': self.get_plugins(),
            'current_user': self.get_current_user(),
            'sd_state': self.get_sd_state()
        }
        
        # Calculate derived information
        if status['printer'] and status['job']:
            printer_data = status['printer']
            job_data = status['job']
            
            # Print progress
            if job_data.get('progress'):
                status['print_progress'] = {
                    'completion': job_data['progress'].get('completion', 0),
                    'print_time': job_data['progress'].get('printTime', 0),
                    'print_time_left': job_data['progress'].get('printTimeLeft', 0),
                    'file_pos': job_data['progress'].get('filepos', 0)
                }
            
            # Temperature summary
            if printer_data.get('temperature'):
                temps = printer_data['temperature']
                status['temperature_summary'] = {
                    'tool_temp': temps.get('tool0', {}).get('actual', 0),
                    'tool_target': temps.get('tool0', {}).get('target', 0),
                    'bed_temp': temps.get('bed', {}).get('actual', 0),
                    'bed_target': temps.get('bed', {}).get('target', 0)
                }
            
            # Printer state
            if printer_data.get('state'):
                state_info = printer_data['state']
                status['printer_state'] = {
                    'text': state_info.get('text', 'Unknown'),
                    'flags': state_info.get('flags', {}),
                    'is_operational': state_info.get('flags', {}).get('operational', False),
                    'is_printing': state_info.get('flags', {}).get('printing', False),
                    'is_paused': state_info.get('flags', {}).get('paused', False),
                    'is_error': state_info.get('flags', {}).get('error', False)
                }
        
        return status
    
    def get_quick_status(self) -> Dict:
        """Get quick status for frequent updates"""
        return {
            'timestamp': time.time(),
            'printer': self.get_printer_status(),
            'job': self.get_job_info(),
            'connection': self.get_connection_info()
        }

# Factory function for easy initialization
def create_octoprint_api(api_key: str, base_url: str, port: int = 80) -> ComprehensiveOctoPrintAPI:
    """Create OctoPrint API instance"""
    config = OctoPrintConfig(
        api_key=api_key,
        base_url=base_url,
        port=port
    )
    return ComprehensiveOctoPrintAPI(config)

if __name__ == "__main__":
    # Test the API
    api = create_octoprint_api(
        api_key="09C668315A784B138FF05305A5DF4E3F",
        base_url="http://192.168.1.13"
    )
    
    # Test comprehensive status
    status = api.get_comprehensive_status()
    print("Comprehensive Status:")
    print(json.dumps(status, indent=2, default=str))
