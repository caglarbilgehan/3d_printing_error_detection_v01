"""
Configuration management for 3D Printer Monitoring System
Handles system-wide settings (detection, performance, etc.)
Printer-specific settings are managed by printers_config.py
"""
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

DEFAULT_CONFIG = {
    'system': {
        'language': 'tr',
        'theme': 'dark',
        'auto_start_detection': True
    },
    'detection': {
        'frame_skip': 2,
        'jpeg_quality': 85,
        'resize_factor': 1.0,
        'motion_threshold': 25,
        'error_sensitivity': 0.5
    },
    'performance': {
        'max_fps': 30,
        'enable_gpu': False,
        'buffer_size': 10
    },
    # Legacy support - will be migrated to printers_config
    'octoprint': {
        'url': 'http://192.168.1.13',
        'port': 80,
        'api_key': '',
        'camera_url': '/webcam/?action=stream'
    }
}


def load_config():
    """Load configuration from file, create with defaults if not exists"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged = DEFAULT_CONFIG.copy()
                for key in merged:
                    if key in config:
                        if isinstance(merged[key], dict):
                            merged[key].update(config[key])
                        else:
                            merged[key] = config[key]
                return merged
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Config] Error loading config: {e}, using defaults")
            return DEFAULT_CONFIG.copy()
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"[Config] Error saving config: {e}")
        return False


def get_detection_settings():
    """Get detection settings"""
    config = load_config()
    return config.get('detection', DEFAULT_CONFIG['detection'])


def update_detection_settings(**kwargs):
    """Update detection settings"""
    config = load_config()
    for key, value in kwargs.items():
        if key in config['detection']:
            config['detection'][key] = value
    return save_config(config)


# ==================== LEGACY FUNCTIONS ====================
# These are kept for backward compatibility during migration
# They should be replaced with printers_config.py functions

def get_octoprint_url():
    """Get full OctoPrint URL with port - LEGACY, use printers_config instead"""
    config = load_config()
    url = config['octoprint']['url'].rstrip('/')
    port = config['octoprint']['port']
    if port and port != 80:
        return f"{url}:{port}"
    return url


def get_octoprint_api_key():
    """Get OctoPrint API key - LEGACY, use printers_config instead"""
    config = load_config()
    return config['octoprint']['api_key']


def get_camera_url():
    """Get full camera stream URL - LEGACY, use printers_config instead"""
    config = load_config()
    base_url = get_octoprint_url()
    camera_path = config['octoprint']['camera_url']
    if camera_path.startswith('http'):
        return camera_path
    return f"{base_url}{camera_path}"


def update_octoprint_settings(url=None, port=None, api_key=None, camera_url=None):
    """Update OctoPrint settings - LEGACY, use printers_config instead"""
    config = load_config()
    if url is not None:
        config['octoprint']['url'] = url.rstrip('/')
    if port is not None:
        config['octoprint']['port'] = int(port) if port else 80
    if api_key is not None:
        config['octoprint']['api_key'] = api_key
    if camera_url is not None:
        config['octoprint']['camera_url'] = camera_url
    return save_config(config)
