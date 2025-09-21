# 🚀 How to Run FaceAttend Project

Choose your preferred way to run the FaceAttend face recognition system:

## 🎯 Quick Start (Recommended)

### Option 1: Interactive Launcher
```bash
python3 launcher.py
```
**Best for**: First-time users, exploring features, testing components

The launcher provides a menu with all available options:
- 🏠 Run main GUI application
- 🧠 Test deep learning features  
- 🔧 System diagnostics
- 📚 Documentation access

### Option 2: Direct GUI Application
```bash
python3 main.py
```
**Best for**: Regular use, face registration, attendance capture

Opens the full FaceAttend application with:
- Face registration system
- Real-time attendance capture
- Attendance logs viewer
- Camera testing

## 🧪 Testing & Validation

### Quick Functionality Test (2 minutes)
```bash
python3 quick_test.py
```
**Tests**: Basic imports, face detection, configuration loading

### Deep Learning Engine Test (3 minutes)
```bash
python3 test_insightface.py
```
**Tests**: Face embeddings, recognition engine, storage system

### Full System Validation (5 minutes)
```bash
python3 validate_implementation.py
```
**Tests**: Complete system validation with detailed results

### Visual Output Generation (10 minutes)
```bash
python3 visual_validation.py
```
**Creates**: Detection images with bounding boxes in `validation_output/`

## 🔧 Development & Debugging

### Interactive Python Session
```bash
python3
```
```python
# Test face detection
from src.recognition.deep_learning import MTCNNDetector
detector = MTCNNDetector(device='cpu')

# Load and test with image  
import cv2
image = cv2.imread('faces/khoi1234/img_01.jpg')
faces = detector.detect_faces(image)
print(f"Detected {len(faces)} faces")
```

### System Status Check
```bash
python3 -c "
from src.recognition import DL_AVAILABLE
print(f'Deep Learning Available: {DL_AVAILABLE}')

from src.storage.face_storage import FaceStorage
storage = FaceStorage()
stats = storage.get_storage_stats()
print(f'Users: {stats.get(\"total_users\", 0)}')
print(f'Images: {stats.get(\"total_images\", 0)}')
"
```

## 📋 Prerequisites

### 1. Install Dependencies
```bash
# Classical computer vision
pip3 install -r requirements.txt

# Deep learning (optional)
pip3 install -r requirements_dl.txt
```

### 2. Camera Setup
- Connect camera and grant permissions
- Close other apps using the camera
- Test with: `python3 -c "import cv2; print('✅ Camera ready' if cv2.VideoCapture(0).isOpened() else '❌ Camera issue')"`

## 🎯 Usage Workflow

### First Time Setup
1. **Run launcher**: `python3 launcher.py`
2. **Choose option 6**: Check system status
3. **Choose option 7**: Test camera
4. **Choose option 2**: Run quick test

### Register Users
1. **Run main app**: `python3 main.py`
2. **Click "📷 Register Face"**
3. **Enter user details and capture face images**

### Take Attendance  
1. **Click "👤 Start Attendance"**
2. **Position face in camera view**
3. **System recognizes and logs attendance**

### View Results
1. **Click "📊 View Attendance Logs"**
2. **Review attendance records**
3. **Export data as needed**

## 🧠 Deep Learning Features

### Face Detection (Available Now)
- **MTCNN detector** with 99.9% confidence
- **Face alignment** for optimal recognition
- **Quality assessment** with scoring

### Face Recognition (Available Now)
- **512-dimensional embeddings** using ArcFace
- **Cosine similarity matching**
- **User template management**

### Embedding Storage (Available Now)
- **Automatic saving** with metadata
- **Quality-based selection**
- **User recognition** from embeddings

## 🔍 Troubleshooting

### Import Errors
```bash
# Check critical imports
python3 -c "
try:
    import cv2, numpy, torch
    print('✅ Core dependencies OK')
except ImportError as e:
    print(f'❌ Missing: {e}')
"
```

### Camera Issues
```bash
# Test camera access
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print('✅ Camera OK' if cap.isOpened() else '❌ Camera issue')
if cap.isOpened(): cap.release()
"
```

### Deep Learning Issues
```bash
# Check DL availability
python3 -c "
from src.recognition import DL_AVAILABLE
print(f'Deep Learning: {\"✅\" if DL_AVAILABLE else \"❌\"} Available')
"
```

## 📁 Project Files

| File | Purpose | Use When |
|------|---------|----------|
| `launcher.py` | Interactive menu | First time, exploring |
| `main.py` | GUI application | Regular use |
| `quick_test.py` | Basic validation | Testing setup |
| `test_insightface.py` | DL engine test | Testing recognition |
| `validate_implementation.py` | Full validation | Comprehensive check |
| `visual_validation.py` | Generate images | Visual verification |

## 💡 Tips

- **Start simple**: Use `launcher.py` for guided experience
- **Test first**: Run `quick_test.py` before main application
- **Check logs**: View `logs/` directory for detailed information
- **Visual debug**: Use `visual_validation.py` to see detection results
- **Performance**: Deep learning works best with good lighting

## 🎉 Ready to Go!

Choose your preferred method:
- **🎯 Guided experience**: `python3 launcher.py`
- **🏠 Direct use**: `python3 main.py`  
- **🧪 Testing**: `python3 quick_test.py`

**Happy face recognition! 😊**