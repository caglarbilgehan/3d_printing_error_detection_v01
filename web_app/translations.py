"""
Multi-language support for 3D Printer Monitoring System
"""

TRANSLATIONS = {
    'tr': {
        # Navigation
        'nav_dashboard': 'Kontrol Paneli',
        'nav_octoprint': 'OctoPrint Kontrolü',
        'nav_roi': 'ROI Ayarları',
        'nav_analytics': 'Analitik',
        'nav_history': 'Geçmiş',
        'nav_documentation': 'Dokümantasyon',
        
        # Header
        'header_title': '3D Yazıcı İzleme',
        'settings': 'Ayarlar',
        'roi_config': 'ROI Yapılandırması',
        'performance_settings': 'Performans Ayarları',
        'camera_settings': 'Kamera Ayarları',
        'detection_settings': 'Tespit Ayarları',
        'logout': 'Çıkış',
        'language': 'Dil',
        
        # Sidebar - Error Detection
        'error_detection': 'Hata Tespiti',
        'active_errors': 'Aktif Hatalar',
        'separation': 'Ayrılma',
        'underextrusion': 'Eksik Akış',
        'deformation': 'Deformasyon',
        'surface_defect': 'Yüzey',
        'model_deviation': 'Sapma',
        'baseline_creating': 'Baseline oluşturuluyor...',
        'baseline_ready': 'Baseline oluşturuldu',
        
        # Sidebar - System Info
        'system_info': 'Sistem Bilgisi',
        'cpu_usage': 'CPU Kullanımı',
        'memory_usage': 'Bellek Kullanımı',
        
        # Dashboard
        'dashboard': 'Kontrol Paneli',
        'status': 'Durum',
        'motion_ratio': 'Hareket Oranı',
        'uptime': 'Çalışma Süresi',
        'frames': 'Kareler',
        'printing': 'BASILIYOR',
        'idle': 'BEKLEMEDE',
        
        # Video Feeds
        'image_processing': 'Görüntü İşleme Hattı',
        'original_feed': 'Orijinal Kamera',
        'motion_mask': 'Hareket Maskesi',
        'motion_history': 'Hareket Geçmişi',
        'roi_mask': 'ROI Maskesi',
        'error_masks': 'Hata Maskeleri',
        
        # OctoPrint
        'octoprint_integration': 'OctoPrint Entegrasyonu',
        'nozzle': 'Nozul',
        'bed': 'Tabla',
        'target': 'Hedef',
        'current_job': 'Aktif İş',
        'no_active_job': 'Aktif iş yok',
        'time_left': 'Kalan Süre',
        'full_control': 'Tam Kontrol Paneli',
        
        # Error Types
        'error_separation_name': 'Ayrılma (Warping)',
        'error_separation_desc': 'Nesnenin baskı yatağından kalkması',
        'error_underextrusion_name': 'Eksik Malzeme Akışı',
        'error_underextrusion_desc': 'Nozuldan filament çıkmaması',
        'error_deformation_name': 'Deforme Olmuş Nesne',
        'error_deformation_desc': 'Şeklin CAD modeline uymaması',
        'error_surface_name': 'Yüzey Hataları',
        'error_surface_desc': 'Yüzeyin CAD modelinden sapması',
        'error_deviation_name': 'Modelden Sapma',
        'error_deviation_desc': 'Yapı/boyut olarak modelden sapma',
        
        # Buttons
        'save': 'Kaydet',
        'cancel': 'İptal',
        'apply': 'Uygula',
        'reset': 'Sıfırla',
        'start': 'Başlat',
        'pause': 'Duraklat',
        'resume': 'Devam',
        'stop': 'Durdur',
        
        # Footer
        'footer_rights': '2024 3D Yazıcı İzleme Sistemi. Tüm hakları saklıdır.',
        'documentation': 'Dokümantasyon',
        'support': 'Destek',
        
        # ROI Setup
        'roi_setup': 'ROI Kurulumu - 3D Baskı Alanı Tespiti',
        'roi_instructions': 'Talimatlar',
        'roi_inst_1': 'Video üzerine tıklayarak noktalar ekleyin (minimum 3 gerekli)',
        'roi_inst_2': 'Noktalar 3D baskı alanınızın etrafında bir çokgen oluşturacak',
        'roi_inst_3': 'Bittiğinde "ROI Kaydet" butonuna tıklayın',
        'roi_controls': 'ROI Kontrolleri',
        'save_roi': 'ROI Kaydet',
        'undo_point': 'Son Noktayı Geri Al',
        'clear_all': 'Tümünü Temizle',
        'selected_points': 'Seçili Noktalar',
        'roi_saved': 'ROI başarıyla kaydedildi!',
        
        # Documentation
        'doc_title': 'Sistem Dokümantasyonu',
        'doc_overview': 'Genel Bakış',
        'doc_image_processing': 'Görüntü İşleme',
        'doc_error_detection': 'Hata Tespiti',
        'doc_motion_analysis': 'Hareket Analizi',
        'doc_algorithms': 'Kullanılan Algoritmalar',
    },
    'en': {
        # Navigation
        'nav_dashboard': 'Dashboard',
        'nav_octoprint': 'OctoPrint Control',
        'nav_roi': 'ROI Setup',
        'nav_analytics': 'Analytics',
        'nav_history': 'History',
        'nav_documentation': 'Documentation',
        
        # Header
        'header_title': '3D Printer Monitor',
        'settings': 'Settings',
        'roi_config': 'ROI Configuration',
        'performance_settings': 'Performance Settings',
        'camera_settings': 'Camera Settings',
        'detection_settings': 'Detection Settings',
        'logout': 'Logout',
        'language': 'Language',
        
        # Sidebar - Error Detection
        'error_detection': 'Error Detection',
        'active_errors': 'Active Errors',
        'separation': 'Separation',
        'underextrusion': 'Under-extrusion',
        'deformation': 'Deformation',
        'surface_defect': 'Surface',
        'model_deviation': 'Deviation',
        'baseline_creating': 'Creating baseline...',
        'baseline_ready': 'Baseline ready',
        
        # Sidebar - System Info
        'system_info': 'System Info',
        'cpu_usage': 'CPU Usage',
        'memory_usage': 'Memory Usage',
        
        # Dashboard
        'dashboard': 'Dashboard',
        'status': 'Status',
        'motion_ratio': 'Motion Ratio',
        'uptime': 'Uptime',
        'frames': 'Frames',
        'printing': 'PRINTING',
        'idle': 'IDLE',
        
        # Video Feeds
        'image_processing': 'Image Processing Pipeline',
        'original_feed': 'Original Camera',
        'motion_mask': 'Motion Mask',
        'motion_history': 'Motion History',
        'roi_mask': 'ROI Mask',
        'error_masks': 'Error Masks',
        
        # OctoPrint
        'octoprint_integration': 'OctoPrint Integration',
        'nozzle': 'Nozzle',
        'bed': 'Bed',
        'target': 'Target',
        'current_job': 'Current Job',
        'no_active_job': 'No active job',
        'time_left': 'Time Left',
        'full_control': 'Full Control Panel',
        
        # Error Types
        'error_separation_name': 'Separation (Warping)',
        'error_separation_desc': 'Object lifting from print bed',
        'error_underextrusion_name': 'Under-extrusion',
        'error_underextrusion_desc': 'Filament not extruding from nozzle',
        'error_deformation_name': 'Deformed Object',
        'error_deformation_desc': 'Shape not matching CAD model',
        'error_surface_name': 'Surface Defects',
        'error_surface_desc': 'Surface deviating from CAD model',
        'error_deviation_name': 'Model Deviation',
        'error_deviation_desc': 'Structure/size deviation from model',
        
        # Buttons
        'save': 'Save',
        'cancel': 'Cancel',
        'apply': 'Apply',
        'reset': 'Reset',
        'start': 'Start',
        'pause': 'Pause',
        'resume': 'Resume',
        'stop': 'Stop',
        
        # Footer
        'footer_rights': '2024 3D Printer Monitoring System. All rights reserved.',
        'documentation': 'Documentation',
        'support': 'Support',
        
        # ROI Setup
        'roi_setup': 'ROI Setup - 3D Print Area Detection',
        'roi_instructions': 'Instructions',
        'roi_inst_1': 'Click on video to add points (minimum 3 required)',
        'roi_inst_2': 'Points will form a polygon around your 3D print area',
        'roi_inst_3': 'Click "Save ROI" when done',
        'roi_controls': 'ROI Controls',
        'save_roi': 'Save ROI',
        'undo_point': 'Undo Last Point',
        'clear_all': 'Clear All',
        'selected_points': 'Selected Points',
        'roi_saved': 'ROI saved successfully!',
        
        # Documentation
        'doc_title': 'System Documentation',
        'doc_overview': 'Overview',
        'doc_image_processing': 'Image Processing',
        'doc_error_detection': 'Error Detection',
        'doc_motion_analysis': 'Motion Analysis',
        'doc_algorithms': 'Algorithms Used',
    }
}

def get_translation(key, lang='tr'):
    """Get translation for a key in specified language"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['tr']).get(key, key)

def get_all_translations(lang='tr'):
    """Get all translations for a language"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['tr'])
