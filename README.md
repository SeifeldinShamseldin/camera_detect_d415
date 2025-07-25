# Advanced Multi-Modal Object Detection with 6D Pose Estimation

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.5+-green.svg)
![RealSense](https://img.shields.io/badge/realsense-D415-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An industrial-grade object detection and 6D pose estimation system optimized for Intel RealSense D415 cameras. Features multi-modal sensor fusion, advanced feature matching, and real-time pose tracking with millimeter precision.

## 🚀 Features

### Multi-Modal Detection (9 Modalities)
- **SIFT Features** (25% weight) - Scale-invariant feature matching
- **AKAZE Features** (15% weight) - Accelerated-KAZE for robust detection  
- **KAZE Features** (15% weight) - Nonlinear scale space detection
- **RGB Template Matching** (15% weight) - Direct template correlation
- **Depth Analysis** (10% weight) - 3D structure validation
- **IR Signature** (8% weight) - Thermal/infrared matching
- **Contour Matching** (8% weight) - Shape-based detection
- **Edge Matching** (2% weight) - Edge pattern recognition
- **Complex Features** (2% weight) - Advanced geometric features

### 6D Pose Estimation
- **Position tracking**: X, Y, Z coordinates with ±2mm precision
- **Orientation tracking**: Roll, Pitch, Yaw angles  
- **Temporal smoothing**: Industrial-grade stability filtering
- **Point cloud analysis**: Centroid-based pose calculation
- **Real-time performance**: 30+ FPS with pose estimation

### Industrial Features
- **Template persistence**: Save/load templates across sessions
- **False positive prevention**: Single master threshold (0.75)
- **Weighted fusion**: Balanced multi-modal decision making
- **Far-distance detection**: Effective up to 80+ cm range
- **Noise filtering**: Statistical outlier removal

## 📋 Requirements

### Hardware
- **Intel RealSense D415** camera (required)
- **USB 3.0** port for camera connection
- **4GB+ RAM** for real-time processing

### Software Dependencies

```bash
# Core dependencies
pip install opencv-python>=4.5.0
pip install numpy>=1.20.0
pip install pyrealsense2>=2.50.0

# Optional advanced features
pip install scipy>=1.7.0
pip install scikit-image>=0.18.0
```

### System Requirements
- **Python 3.8+**
- **Linux** (Ubuntu 18.04+) or **Windows 10+**
- **OpenCV** with xfeatures2d (for SIFT/SURF)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/SeifeldinShamseldin/camera_detect_d415.git
cd camera_detect_d415
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Connect RealSense D415 camera**
```bash
# Verify camera connection
rs-enumerate-devices
```

4. **Run the detection system**
```bash
python3 test_robust_detection.py
```

## 🎯 Usage Guide

### Basic Operation

1. **Start the system**
```bash
python3 test_robust_detection.py
```

2. **Create templates**
   - Press `'t'` to enter template creation mode
   - Click and drag to select object ROI
   - System automatically extracts 9-modal features
   - Template saved with 6D pose reference data

3. **Real-time detection**
   - System detects objects using all 9 modalities
   - Displays confidence scores and best detection method
   - Shows 6D pose: `X:0.025 Y:-0.012 Z:0.341 Roll:0.0° Pitch:0.0° Yaw:15.3°`

4. **Exit system**
   - Press `ESC` to quit

### Advanced Configuration

#### Adjust Detection Sensitivity
```python
# In test_robust_detection.py
self.MASTER_THRESHOLD = 0.75  # Higher = stricter (0.60-0.90)
```

#### Modify Modal Weights
```python
self.weights = {
    'sift': 0.30,           # Increase SIFT importance
    'rgb_template': 0.20,   # Increase template matching
    'depth': 0.15,          # Increase depth validation
    # ... adjust other weights
}
```

#### 6D Pose Parameters  
```python
self.position_threshold = 0.005   # 5mm movement threshold
self.smoothing_factor = 0.3       # Temporal smoothing (0.1-0.5)
```

## 📊 Detection Performance

| Distance | Point Cloud Size | Detection Confidence | Pose Precision |
|----------|------------------|---------------------|----------------|
| 30-40cm  | 50,000+ points   | 94-100%            | ±1-2mm         |
| 40-60cm  | 24,000-36,000    | 93-99%             | ±2-5mm         |
| 60-80cm  | 3,600-15,000     | 66-94%             | ±5-10mm        |
| 80+cm    | 1,000-3,600      | 60-80%             | ±10-20mm       |

## 🏗️ System Architecture

```
┌─────────────────────┐
│   RealSense D415    │
│  (RGB+Depth+IR)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Feature Extraction  │
│ - SIFT/AKAZE/KAZE  │
│ - RGB/Depth/IR     │
│ - Contours/Edges   │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Weighted Fusion     │
│ Single Threshold    │
│ (0.75 confidence)   │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  6D Pose Estimation │
│  Point Cloud ICP    │
│  Temporal Smoothing │
└─────────────────────┘
```

## 📁 File Structure

```
camera_detect_d415/
├── test_robust_detection.py    # Main detection system
├── robust_templates/           # Template storage directory
│   └── template_name/
│       ├── rgb.jpg            # RGB template image
│       ├── depth.npy          # Depth data
│       ├── ir.jpg             # IR template
│       ├── pointcloud.npy     # Point cloud data
│       ├── pose_centroid.npy  # 6D pose reference
│       ├── pose_pointcloud.npy # Pose point cloud
│       └── features.json      # Feature descriptors
├── README.md                  # This documentation
└── requirements.txt           # Python dependencies
```

## 🔧 Troubleshooting

### Camera Issues
```bash
# Check camera connection
rs-enumerate-devices

# Fix permissions (Linux)
sudo usermod -a -G plugdev $USER
sudo udevadm control --reload-rules
```

### Performance Issues
- **Reduce point cloud density**: Increase `downsample_factor`
- **Lower detection rate**: Add frame skipping
- **Disable 6D pose**: Comment out pose estimation code

### False Positives
- **Increase threshold**: `MASTER_THRESHOLD = 0.80`
- **Reduce SIFT weight**: `'sift': 0.15`
- **Add size validation**: Adjust min/max object sizes

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Seifeddin Shamseldin**
- GitHub: [@SeifeldinShamseldin](https://github.com/SeifeldinShamseldin)
- Email: seifeddin.shamseldin@example.com

## 🙏 Acknowledgments

- Intel RealSense team for excellent camera SDK
- OpenCV community for computer vision tools
- Industrial robotics applications that inspired this work

## 📈 Roadmap

- [ ] Add more camera support (D435, L515)
- [ ] Implement deep learning features
- [ ] Add calibration tools
- [ ] Create Python package
- [ ] Add ROS integration
- [ ] Build GUI interface

---

⭐ **Star this repository if it helped your project!**