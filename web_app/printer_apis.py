"""
Multi-Printer API Integration Module
Supports: OctoPrint, Repetier Server, Klipper/Moonraker, Duet
"""

import requests
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

class PrinterState(Enum):
    OFFLINE = "offline"
    IDLE = "idle"
    PRINTING = "printing"
    PAUSED = "paused"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class PrinterStatus:
    """Unified printer status across all printer types"""
    state: PrinterState
    state_text: str
    nozzle_temp: float = 0.0
    nozzle_target: float = 0.0
    bed_temp: float = 0.0
    bed_target: float = 0.0
    progress: float = 0.0
    print_time: int = 0  # seconds
    print_time_left: int = 0  # seconds
    current_file: Optional[str] = None
    current_layer: int = 0
    total_layers: int = 0
    z_height: float = 0.0
    fan_speed: int = 0  # 0-100
    feed_rate: int = 100  # percentage
    flow_rate: int = 100  # percentage
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'state_text': self.state_text,
            'nozzle_temp': self.nozzle_temp,
            'nozzle_target': self.nozzle_target,
            'bed_temp': self.bed_temp,
            'bed_target': self.bed_target,
            'progress': self.progress,
            'print_time': self.print_time,
            'print_time_left': self.print_time_left,
            'current_file': self.current_file,
            'current_layer': self.current_layer,
            'total_layers': self.total_layers,
            'z_height': self.z_height,
            'fan_speed': self.fan_speed,
            'feed_rate': self.feed_rate,
            'flow_rate': self.flow_rate
        }


class PrinterAPI(ABC):
    """Abstract base class for printer APIs"""
    
    def __init__(self, host: str, port: int, api_key: str = "", ssl: bool = False, timeout: int = 10):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.ssl = ssl
        self.timeout = timeout
        self.base_url = f"{'https' if ssl else 'http'}://{host}:{port}"
    
    @abstractmethod
    def get_status(self) -> PrinterStatus:
        """Get current printer status"""
        pass
    
    @abstractmethod
    def get_files(self) -> List[Dict]:
        """Get list of files"""
        pass
    
    @abstractmethod
    def start_print(self, filename: str) -> bool:
        """Start printing a file"""
        pass
    
    @abstractmethod
    def pause_print(self) -> bool:
        """Pause current print"""
        pass
    
    @abstractmethod
    def resume_print(self) -> bool:
        """Resume paused print"""
        pass
    
    @abstractmethod
    def cancel_print(self) -> bool:
        """Cancel current print"""
        pass
    
    @abstractmethod
    def set_temperature(self, tool: str, target: float) -> bool:
        """Set temperature for tool (nozzle/bed)"""
        pass
    
    @abstractmethod
    def send_gcode(self, command: str) -> bool:
        """Send G-code command"""
        pass
    
    @abstractmethod
    def home(self, axes: List[str] = None) -> bool:
        """Home specified axes"""
        pass
    
    @abstractmethod
    def jog(self, x: float = 0, y: float = 0, z: float = 0, speed: int = 1000) -> bool:
        """Jog printer head"""
        pass
    
    def is_connected(self) -> bool:
        """Check if printer is connected"""
        try:
            status = self.get_status()
            return status.state != PrinterState.OFFLINE
        except:
            return False


class OctoPrintAPI(PrinterAPI):
    """OctoPrint API Implementation"""
    
    def _request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Optional[Dict]:
        """Make API request to OctoPrint"""
        try:
            headers = {'X-Api-Key': self.api_key}
            url = f"{self.base_url}/api/{endpoint}"
            
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=self.timeout)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            else:
                return None
            
            if response.status_code == 200:
                return response.json() if response.text else {}
            elif response.status_code == 204:
                return {}
            return None
        except Exception as e:
            print(f"OctoPrint API error: {e}")
            return None
    
    def get_status(self) -> PrinterStatus:
        """Get OctoPrint status"""
        status = PrinterStatus(
            state=PrinterState.OFFLINE,
            state_text="Offline"
        )
        
        try:
            # Get printer state
            printer_data = self._request('printer')
            if printer_data:
                state_data = printer_data.get('state', {})
                flags = state_data.get('flags', {})
                
                if flags.get('printing'):
                    status.state = PrinterState.PRINTING
                    status.state_text = "Printing"
                elif flags.get('paused') or flags.get('pausing'):
                    status.state = PrinterState.PAUSED
                    status.state_text = "Paused"
                elif flags.get('ready') or flags.get('operational'):
                    status.state = PrinterState.IDLE
                    status.state_text = "Idle"
                elif flags.get('error'):
                    status.state = PrinterState.ERROR
                    status.state_text = "Error"
                
                # Temperatures
                temps = printer_data.get('temperature', {})
                tool0 = temps.get('tool0', {})
                bed = temps.get('bed', {})
                
                status.nozzle_temp = tool0.get('actual', 0)
                status.nozzle_target = tool0.get('target', 0)
                status.bed_temp = bed.get('actual', 0)
                status.bed_target = bed.get('target', 0)
            
            # Get job info
            job_data = self._request('job')
            if job_data:
                progress = job_data.get('progress', {})
                job = job_data.get('job', {})
                
                status.progress = progress.get('completion', 0) or 0
                status.print_time = progress.get('printTime', 0) or 0
                status.print_time_left = progress.get('printTimeLeft', 0) or 0
                
                file_info = job.get('file', {})
                status.current_file = file_info.get('name')
                
        except Exception as e:
            print(f"Error getting OctoPrint status: {e}")
        
        return status
    
    def get_files(self) -> List[Dict]:
        """Get files from OctoPrint"""
        result = self._request('files')
        if result:
            return result.get('files', [])
        return []
    
    def start_print(self, filename: str) -> bool:
        """Start print on OctoPrint"""
        result = self._request(f'files/local/{filename}', 'POST', {'command': 'select', 'print': True})
        return result is not None
    
    def pause_print(self) -> bool:
        """Pause print on OctoPrint"""
        result = self._request('job', 'POST', {'command': 'pause', 'action': 'pause'})
        return result is not None
    
    def resume_print(self) -> bool:
        """Resume print on OctoPrint"""
        result = self._request('job', 'POST', {'command': 'pause', 'action': 'resume'})
        return result is not None
    
    def cancel_print(self) -> bool:
        """Cancel print on OctoPrint"""
        result = self._request('job', 'POST', {'command': 'cancel'})
        return result is not None
    
    def set_temperature(self, tool: str, target: float) -> bool:
        """Set temperature on OctoPrint"""
        if tool == 'bed':
            result = self._request('printer/bed', 'POST', {'command': 'target', 'target': target})
        else:
            result = self._request('printer/tool', 'POST', {'command': 'target', 'targets': {'tool0': target}})
        return result is not None
    
    def send_gcode(self, command: str) -> bool:
        """Send G-code to OctoPrint"""
        result = self._request('printer/command', 'POST', {'commands': [command]})
        return result is not None
    
    def home(self, axes: List[str] = None) -> bool:
        """Home axes on OctoPrint"""
        if axes is None:
            axes = ['x', 'y', 'z']
        result = self._request('printer/printhead', 'POST', {'command': 'home', 'axes': axes})
        return result is not None
    
    def jog(self, x: float = 0, y: float = 0, z: float = 0, speed: int = 1000) -> bool:
        """Jog printer on OctoPrint"""
        data = {'command': 'jog', 'speed': speed}
        if x != 0:
            data['x'] = x
        if y != 0:
            data['y'] = y
        if z != 0:
            data['z'] = z
        result = self._request('printer/printhead', 'POST', data)
        return result is not None


class RepetierAPI(PrinterAPI):
    """Repetier Server API Implementation"""
    
    def _request(self, action: str, data: Dict = None) -> Optional[Dict]:
        """Make API request to Repetier Server"""
        try:
            url = f"{self.base_url}/printer/api/{self.api_key}"
            params = {'a': action}
            if data:
                params['data'] = json.dumps(data)
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Repetier API error: {e}")
            return None
    
    def get_status(self) -> PrinterStatus:
        """Get Repetier Server status"""
        status = PrinterStatus(
            state=PrinterState.OFFLINE,
            state_text="Offline"
        )
        
        try:
            result = self._request('stateList')
            if result and len(result) > 0:
                printer = result[0]  # First printer
                
                if printer.get('online'):
                    if printer.get('job'):
                        status.state = PrinterState.PRINTING
                        status.state_text = "Printing"
                    elif printer.get('paused'):
                        status.state = PrinterState.PAUSED
                        status.state_text = "Paused"
                    else:
                        status.state = PrinterState.IDLE
                        status.state_text = "Idle"
                
                # Temperatures
                extruder = printer.get('extruder', [{}])[0] if printer.get('extruder') else {}
                status.nozzle_temp = extruder.get('tempRead', 0)
                status.nozzle_target = extruder.get('tempSet', 0)
                
                bed = printer.get('heatedBed', {})
                status.bed_temp = bed.get('tempRead', 0)
                status.bed_target = bed.get('tempSet', 0)
                
                # Job info
                job = printer.get('job', '')
                if job:
                    status.current_file = job
                    status.progress = printer.get('done', 0)
                    
                # Layer info
                status.current_layer = printer.get('layer', 0)
                status.total_layers = printer.get('ofLayer', 0)
                status.z_height = printer.get('z', 0)
                
                # Speed/Flow
                status.feed_rate = printer.get('speedMultiply', 100)
                status.flow_rate = printer.get('flowMultiply', 100)
                status.fan_speed = printer.get('fanPercent', 0)
                
        except Exception as e:
            print(f"Error getting Repetier status: {e}")
        
        return status
    
    def get_files(self) -> List[Dict]:
        """Get files from Repetier"""
        result = self._request('listModels')
        if result:
            return result.get('data', [])
        return []
    
    def start_print(self, filename: str) -> bool:
        result = self._request('copyModel', {'id': filename})
        return result is not None
    
    def pause_print(self) -> bool:
        result = self._request('pauseJob')
        return result is not None
    
    def resume_print(self) -> bool:
        result = self._request('continueJob')
        return result is not None
    
    def cancel_print(self) -> bool:
        result = self._request('stopJob')
        return result is not None
    
    def set_temperature(self, tool: str, target: float) -> bool:
        if tool == 'bed':
            result = self._request('setHeatedBed', {'temperature': target})
        else:
            result = self._request('setExtruderTemperature', {'temperature': target, 'extruder': 0})
        return result is not None
    
    def send_gcode(self, command: str) -> bool:
        result = self._request('send', {'cmd': command})
        return result is not None
    
    def home(self, axes: List[str] = None) -> bool:
        return self.send_gcode('G28')
    
    def jog(self, x: float = 0, y: float = 0, z: float = 0, speed: int = 1000) -> bool:
        commands = ['G91']  # Relative positioning
        if x != 0 or y != 0:
            commands.append(f'G1 X{x} Y{y} F{speed}')
        if z != 0:
            commands.append(f'G1 Z{z} F{speed}')
        commands.append('G90')  # Absolute positioning
        
        for cmd in commands:
            self.send_gcode(cmd)
        return True


class KlipperAPI(PrinterAPI):
    """Klipper/Moonraker API Implementation"""
    
    def _request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Optional[Dict]:
        """Make API request to Moonraker"""
        try:
            url = f"{self.base_url}/{endpoint}"
            
            if method == 'GET':
                response = requests.get(url, timeout=self.timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Klipper/Moonraker API error: {e}")
            return None
    
    def get_status(self) -> PrinterStatus:
        """Get Klipper status via Moonraker"""
        status = PrinterStatus(
            state=PrinterState.OFFLINE,
            state_text="Offline"
        )
        
        try:
            # Get printer objects
            result = self._request('printer/objects/query', 'POST', {
                'objects': {
                    'print_stats': None,
                    'heater_bed': None,
                    'extruder': None,
                    'fan': None,
                    'gcode_move': None,
                    'virtual_sdcard': None
                }
            })
            
            if result and 'result' in result:
                data = result['result'].get('status', {})
                
                # Print state
                print_stats = data.get('print_stats', {})
                state = print_stats.get('state', 'standby')
                
                if state == 'printing':
                    status.state = PrinterState.PRINTING
                    status.state_text = "Printing"
                elif state == 'paused':
                    status.state = PrinterState.PAUSED
                    status.state_text = "Paused"
                elif state == 'complete':
                    status.state = PrinterState.IDLE
                    status.state_text = "Complete"
                elif state == 'error':
                    status.state = PrinterState.ERROR
                    status.state_text = "Error"
                else:
                    status.state = PrinterState.IDLE
                    status.state_text = "Idle"
                
                status.current_file = print_stats.get('filename')
                status.print_time = int(print_stats.get('print_duration', 0))
                
                # Temperatures
                extruder = data.get('extruder', {})
                status.nozzle_temp = extruder.get('temperature', 0)
                status.nozzle_target = extruder.get('target', 0)
                
                bed = data.get('heater_bed', {})
                status.bed_temp = bed.get('temperature', 0)
                status.bed_target = bed.get('target', 0)
                
                # Progress
                vsd = data.get('virtual_sdcard', {})
                status.progress = vsd.get('progress', 0) * 100
                
                # Fan
                fan = data.get('fan', {})
                status.fan_speed = int(fan.get('speed', 0) * 100)
                
                # Position
                gcode_move = data.get('gcode_move', {})
                pos = gcode_move.get('gcode_position', [0, 0, 0, 0])
                if len(pos) >= 3:
                    status.z_height = pos[2]
                
        except Exception as e:
            print(f"Error getting Klipper status: {e}")
        
        return status
    
    def get_files(self) -> List[Dict]:
        """Get files from Moonraker"""
        result = self._request('server/files/list')
        if result:
            return result.get('result', [])
        return []
    
    def start_print(self, filename: str) -> bool:
        result = self._request('printer/print/start', 'POST', {'filename': filename})
        return result is not None
    
    def pause_print(self) -> bool:
        result = self._request('printer/print/pause', 'POST')
        return result is not None
    
    def resume_print(self) -> bool:
        result = self._request('printer/print/resume', 'POST')
        return result is not None
    
    def cancel_print(self) -> bool:
        result = self._request('printer/print/cancel', 'POST')
        return result is not None
    
    def set_temperature(self, tool: str, target: float) -> bool:
        if tool == 'bed':
            gcode = f'SET_HEATER_TEMPERATURE HEATER=heater_bed TARGET={target}'
        else:
            gcode = f'SET_HEATER_TEMPERATURE HEATER=extruder TARGET={target}'
        return self.send_gcode(gcode)
    
    def send_gcode(self, command: str) -> bool:
        result = self._request('printer/gcode/script', 'POST', {'script': command})
        return result is not None
    
    def home(self, axes: List[str] = None) -> bool:
        return self.send_gcode('G28')
    
    def jog(self, x: float = 0, y: float = 0, z: float = 0, speed: int = 1000) -> bool:
        commands = ['G91']
        if x != 0 or y != 0 or z != 0:
            commands.append(f'G1 X{x} Y{y} Z{z} F{speed}')
        commands.append('G90')
        
        for cmd in commands:
            self.send_gcode(cmd)
        return True


def get_printer_api(printer_type: str, host: str, port: int, api_key: str = "", ssl: bool = False) -> PrinterAPI:
    """Factory function to get appropriate printer API"""
    if printer_type == 'octoprint':
        return OctoPrintAPI(host, port, api_key, ssl)
    elif printer_type == 'repetier':
        return RepetierAPI(host, port, api_key, ssl)
    elif printer_type == 'klipper':
        return KlipperAPI(host, port, api_key, ssl)
    else:
        # Default to OctoPrint
        return OctoPrintAPI(host, port, api_key, ssl)
