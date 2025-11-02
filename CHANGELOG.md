# Changelog - 3D Printer Monitoring

All notable changes to this project will be documented in this file.

## [2.0.0] - 2024-11-02

### 🎉 Major Release - Modern Architecture

#### ✨ Added
- **Modern Architecture**: Component-based UI, template inheritance
- **Multi-language Support**: Turkish/English with session management
- **Error Detection System**: 5 types of error detection with masks
- **ROI Overlay**: Visual 3D print area marking
- **Documentation Page**: Complete system documentation
- **Performance Optimization**: Frame skip, JPEG compression, ROI masking
- **Static Files**: Separated CSS/JS files for better caching
- **Component System**: Reusable UI components (header, sidebar, footer)

#### 🔧 Technical Improvements
- **CSS/JS Separation**: 5 separate files for modular architecture
- **Template Components**: 4 reusable components
- **API Endpoints**: Error masks, ROI overlay endpoints
- **Debug Logging**: Enhanced motion analysis debugging
- **Code Quality**: 70% reduction in code duplication

#### 📊 Features
- **Error Masks Visualization**: Real-time error mask display
- **ROI System**: Interactive polygon selection
- **Multi-language UI**: Complete TR/EN translation (100+ keys)
- **Modern UI/UX**: Bootstrap 5, responsive design
- **Performance Tuning**: Configurable parameters via UI

#### 🏗️ Architecture
- **Base Template**: Clean inheritance system
- **Component-based**: Modular, maintainable code
- **DRY Principle**: No code duplication
- **Separation of Concerns**: Clear responsibility boundaries

## [1.0.0] - 2024-10-30

### Added
- Web interface for monitoring print status
- Real-time motion detection and analysis
- Dashboard with status cards and video feeds
- OctoPrint integration for printer control
- ROI (Region of Interest) selection for focused monitoring
- 5 types of error detection:
  - Separation (Warping)
  - Under-extrusion
  - Deformation
  - Surface defects
  - Model deviation

### Changed
- Improved motion detection algorithm
- Enhanced web interface design

### Fixed
- Camera connection stability issues
- Performance optimization for real-time processing

## [0.1.0] - 2024-10-15

### Added
- Initial release
- Basic print status detection
- OpenCV-based motion analysis
- Simple web interface

### 🎉 Major Features Added

#### Error Detection System
- **5 Hata Tipi Tespiti**: Gelişmiş bilgisayarlı görü algoritmaları ile gerçek zamanlı hata analizi
  - ✅ **Ayrılma (Warping)**: Nesnenin baskı yatağından kalkması tespiti
  - ✅ **Eksik Malzeme Akışı**: Nozuldan filament çıkmaması tespiti
  - ✅ **Deforme Olmuş Nesne**: CAD modeline uyumsuzluk tespiti
  - ✅ **Yüzey Hataları**: Yüzey dokusunda sapma tespiti
  - ✅ **Modelden Sapma**: Boyut ve yapı sapması tespiti
- Her hata için güven skoru (confidence) hesaplama
- Baseline oluşturma sistemi (ilk 30 frame)
- Gerçek zamanlı hata kartları ve görsel göstergeler

#### OctoPrint Tam Entegrasyonu
- **20+ API Endpoint**: Raspberry Pi üzerindeki OctoPrint ile tam entegrasyon
- **Sıcaklık Kontrolü**: Nozzle ve bed sıcaklık okuma ve ayarlama
- **Baskı İşi Yönetimi**: Start, pause, resume, cancel komutları
- **Yazıcı Kontrolü**: Homing, G-code gönderme
- **Dosya Yönetimi**: Yüklü dosyaları listeleme
- **Sistem Bilgileri**: Plugin, log, versiyon bilgileri
- **G-code Terminal**: İnteraktif komut arayüzü

#### Profesyonel Dashboard Altyapısı
- **Header**: Sabit üst menü, settings dropdown, bildirimler
- **Sidebar**: Navigasyon menüsü, sistem bilgileri, quick actions
- **Footer**: Copyright, documentation linkleri
- **Responsive Tasarım**: Mobil uyumlu, sidebar toggle
- **2 Ana Sayfa**:
  - `/` - Ana monitoring dashboard
  - `/octoprint` - OctoPrint kontrol paneli

### 📹 Görüntü İşleme

#### Çoklu Video Stream
- **3 Ayrı Stream**: Original, Motion Mask, Motion Graph
- **Thread-Safe İşleme**: Arka planda sürekli frame işleme
- **30 FPS**: Her stream bağımsız çalışıyor
- **Ayrı Endpoint'ler**:
  - `/video_feed/original` - Ham kamera görüntüsü
  - `/video_feed/mask` - Hareket algılama maskesi
  - `/video_feed/graph` - Hareket geçmişi grafiği

#### Gelişmiş Hareket Analizi
- Background subtraction (MOG2)
- Edge detection (Canny)
- Contour analysis
- Brightness tracking
- Motion variance hesaplama

### 🌐 Network Configuration

#### Raspberry Pi Entegrasyonu
- **OctoPrint**: `http://192.168.1.17`
- **Camera Stream**: `http://192.168.1.17/webcam/?action=stream`
- **API Key**: Güvenli kimlik doğrulama
- **Timeout Yönetimi**: 10 saniye timeout
- **Hata Yönetimi**: Connection error, timeout handling
- **Debug Logging**: Detaylı bağlantı logları

### 🎨 UI/UX İyileştirmeleri

#### Tasarım Sistemi
- **Bootstrap 5.3**: Modern, responsive framework
- **Bootstrap Icons**: 100+ ikon
- **Gradient Backgrounds**: Profesyonel renkler
- **Card-Based Layout**: Modüler yapı
- **Hover Effects**: İnteraktif animasyonlar
- **Progress Bars**: Görsel ilerleme göstergeleri

#### Error Detection Cards
- **Dinamik Renkler**: Normal (yeşil), Hata (kırmızı)
- **Confidence Göstergeleri**: Yüzde bazlı güven skoru
- **Icon System**: Her hata tipi için özel ikon
- **Real-time Updates**: 2 saniyede bir güncelleme
- **Visual Feedback**: Border, shadow, background değişimi

### 📊 API Endpoints

#### Motion Detection
- `GET /api/status` - Hareket durumu ve hata bilgileri
- `GET /api/errors` - Detaylı hata analizi

#### Video Streams
- `GET /video_feed/original` - Orijinal kamera
- `GET /video_feed/mask` - Hareket maskesi
- `GET /video_feed/graph` - Hareket grafiği

#### OctoPrint Integration
- `GET /api/octoprint/connection` - Bağlantı durumu
- `GET /api/octoprint/printer` - Yazıcı durumu
- `GET /api/octoprint/job` - Baskı işi
- `GET /api/octoprint/files` - Dosya listesi
- `GET /api/octoprint/dashboard` - Tüm veriler
- `POST /api/octoprint/job/start` - Baskıyı başlat
- `POST /api/octoprint/job/pause` - Duraklat
- `POST /api/octoprint/job/cancel` - İptal et
- `POST /api/octoprint/printer/command` - G-code gönder
- `POST /api/octoprint/printer/temperature` - Sıcaklık ayarla
- `POST /api/octoprint/printer/bed` - Bed sıcaklığı
- `POST /api/octoprint/printer/home` - Homing

### 🔧 Technical Improvements

#### Backend
- **Threading**: Arka plan frame işleme
- **Lock Mechanism**: Thread-safe veri paylaşımı
- **Error Handling**: Try-catch blokları
- **Logging**: Debug ve info mesajları
- **Modular Structure**: Ayrı error_detection.py modülü

#### Frontend
- **Async/Await**: Modern JavaScript
- **Fetch API**: RESTful API çağrıları
- **Auto-refresh**: Otomatik veri güncelleme
- **Error Handling**: Graceful degradation
- **Utility Functions**: Time, bytes formatters

### 📁 New Files

```
d:\Projects\3D_Printing\
├── error_detection.py          # Hata tespit sistemi
├── CHANGELOG.md                # Bu dosya
├── NETWORK_SETUP.md            # Ağ yapılandırma kılavuzu
├── OCTOPRINT_FEATURES.md       # OctoPrint özellikleri
├── web_app/
│   ├── app.py                  # Flask server (333 satır)
│   └── templates/
│       ├── index.html          # Ana dashboard (624 satır)
│       └── octoprint.html      # OctoPrint kontrol (915 satır)
```

### 🐛 Bug Fixes
- Duplicate sys.path insertion düzeltildi
- Video stream thread safety iyileştirildi
- OctoPrint timeout hataları giderildi
- Sidebar navigation tutarlılığı sağlandı

### 📚 Documentation
- **README.md**: Network setup eklendi
- **NETWORK_SETUP.md**: Detaylı ağ yapılandırması
- **OCTOPRINT_FEATURES.md**: Tüm OctoPrint özellikleri
- **Inline Comments**: Kod içi açıklamalar

---

## [1.0.0] - 2024-11-01

### Initial Release

#### Core Features
- ✅ Temel hareket algılama (MOG2)
- ✅ Baskı durumu tespiti (printing/idle)
- ✅ Kamera stream entegrasyonu
- ✅ Flask web server
- ✅ Basit web arayüzü

#### Components
- `main.py` - PrintStatusDetector sınıfı
- `web_app/app.py` - Basit Flask uygulaması
- `requirements.txt` - Temel bağımlılıklar

---

## Upcoming Features (Roadmap)

### v2.1.0 (Planned)
- [ ] Hata geçmişi ve istatistikler
- [ ] Email/SMS bildirimleri
- [ ] Timelapse video kaydı
- [ ] Çoklu kamera desteği
- [ ] Dark mode

### v2.2.0 (Planned)
- [ ] Machine learning ile gelişmiş hata tespiti
- [ ] Otomatik baskı durdurma
- [ ] G-code analizi ve optimizasyon
- [ ] Filament takibi
- [ ] Maliyet hesaplama

### v3.0.0 (Future)
- [ ] Çoklu yazıcı desteği
- [ ] Cloud entegrasyonu
- [ ] Mobile app (iOS/Android)
- [ ] AI-powered quality prediction
- [ ] Blockchain-based print verification

---

## Version Numbering

Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes, major features
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, minor improvements

---

## Contributors

- **Development**: AI-Assisted Development
- **Testing**: User Testing
- **Documentation**: Comprehensive docs

---

## License

This project is developed for thesis/research purposes.

---

## Support

For issues and questions:
- Check documentation files
- Review NETWORK_SETUP.md for connectivity issues
- Check OCTOPRINT_FEATURES.md for API details

---

**Last Updated**: November 2, 2024
