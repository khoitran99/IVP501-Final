# 🚀 How to Run FaceAttend Project

This guide shows you how to run the FaceAttend face recognition attendance system with both classical and deep learning capabilities.

## 📋 Prerequisites

### 1. Python Environment
```bash
# Check Python version (3.8+ required)
python3 --version

# Check if pip is available
pip3 --version
```

### 2. Install Dependencies
```bash
cd "/Users/tranvankhoi/Documents/MSE/Image Video Processing/final-2"

# Install classical dependencies
pip3 install -r requirements.txt

# Install deep learning dependencies
pip3 install -r requirements_dl.txt
```

### 3. Camera Access
- Ensure your camera is connected and working
- Grant camera permissions if prompted
- Close other applications using the camera

## 🎯 Quick Start Options

### Option 1: Run Classical System (Original)
```bash
# Run the original FaceAttend application
python3 main.py
```

### Option 2: Test Deep Learning Components
```bash
# Test deep learning face detection
python3 quick_test.py

# Test InsightFace recognition engine
python3 test_insightface.py

# Run comprehensive validation
python3 validate_implementation.py

# Generate visual results
python3 visual_validation.py
```

### Option 3: Interactive Testing
```bash
# Open Python interactive session
python3

# Test face detection
from src.recognition.deep_learning import MTCNNDetector
detector = MTCNNDetector(device='cpu')
print("MTCNN detector ready!")

# Test with your images
import cv2
image = cv2.imread('faces/khoi1234/img_01.jpg')
faces = detector.detect_faces(image)
print(f"Detected {len(faces)} faces")
```

## 📱 Running the Full Application

### Step 1: Start the Application
```bash
python3 main.py
```

### Step 2: Using the Interface

**Main Window Features:**
- 📷 **Register Face**: Add new users to the system
- 👤 **Start Attendance**: Capture attendance in real-time
- 📊 **View Attendance Logs**: Review attendance records
- 🎥 **Test Camera**: Verify camera functionality

### Step 3: Register Your First User

1. Click **"📷 Register Face"**
2. Enter user details (Name, ID)
3. Position face in camera view
4. Capture multiple face images (10 recommended)
5. System will process and save the face data

### Step 4: Take Attendance

1. Click **"👤 Start Attendance"**
2. Position face in camera view
3. System will recognize and log attendance
4. View results in real-time

## 🧠 Deep Learning Features

### Test Face Detection
```bash
# Test MTCNN face detector
python3 -c "
from src.recognition.deep_learning import MTCNNDetector
import cv2

detector = MTCNNDetector(device='cpu')
image = cv2.imread('faces/khoi1234/img_01.jpg')
faces = detector.detect_faces(image)

for i, face in enumerate(faces):
    print(f'Face {i+1}: confidence={face.confidence:.3f}, quality={face.quality_score:.3f}')
"
```

### Test Face Recognition
```bash
# Test face embedding generation
python3 -c "
from src.recognition.deep_learning import DLFaceRecognizer, MTCNNDetector
import cv2

# Note: Requires model download for full functionality
try:
    detector = MTCNNDetector(device='cpu')
    recognizer = DLFaceRecognizer(model_type='arcface')
    
    image = cv2.imread('faces/khoi1234/img_01.jpg')
    faces = detector.detect_faces(image)
    
    if faces:
        aligned_face = detector.align_face(image, faces[0])
        embedding = recognizer.get_embedding(aligned_face)
        print(f'Generated embedding: {embedding.dimension} dimensions')
    else:
        print('No faces detected')
        
except Exception as e:
    print(f'Deep learning test: {e}')
    print('This is expected if models are not downloaded')
"
```

### Test Embedding Management
```bash
# Test embedding storage system
python3 -c "
from src.recognition.deep_learning import EmbeddingManager, FaceEmbedding
import numpy as np

manager = EmbeddingManager()

# Create test embedding
test_vector = np.random.randn(512).astype(np.float32)
embedding = FaceEmbedding(test_vector, confidence=0.95, model_name='test')

# Save embedding
manager.add_user_embedding('test_user', embedding, 'test.jpg', 'test_model')

# Retrieve embedding
template = manager.get_user_template('test_user')
print(f'Saved and retrieved embedding: {template.dimension} dimensions')

# Clean up
import shutil
from pathlib import Path
if Path('faces/test_user').exists():
    shutil.rmtree('faces/test_user')
    print('Cleaned up test data')
"
```

## 🔧 Model Management

### Available Models
```bash
# List available models
python3 -c "
from src.recognition.deep_learning import get_dl_model_loader

loader = get_dl_model_loader()
models = loader.get_available_models()

print('Available Deep Learning Models:')
for name, config in models.items():
    status = '✅ Downloaded' if config.get('downloaded') else '⚠️  Not downloaded'
    print(f'  {name}: {config[\"description\"]} ({status})')
"
```

### Download Models (Optional)
```bash
# Try to download ArcFace model
python3 -c "
from models.model_downloader import ModelDownloader

downloader = ModelDownloader()
try:
    path = downloader.download_model('recognition', 'arcface_r100')
    print(f'Downloaded to: {path}')
except Exception as e:
    print(f'Download failed: {e}')
    print('Note: URLs may need updating for actual model files')
"
```

## 📊 System Information

### Check System Status
```bash
# Get system information
python3 -c "
from src.recognition import DL_AVAILABLE
from src.storage.face_storage import FaceStorage

print('=== FaceAttend System Status ===')
print(f'Deep Learning Available: {DL_AVAILABLE}')

storage = FaceStorage()
stats = storage.get_storage_stats()
print(f'Registered Users: {stats.get(\"total_users\", 0)}')
print(f'Total Face Images: {stats.get(\"total_images\", 0)}')
print(f'Storage Size: {stats.get(\"total_size_mb\", 0):.1f} MB')

if DL_AVAILABLE:
    from src.recognition.deep_learning import get_dl_model_loader
    loader = get_dl_model_loader()
    loader_stats = loader.get_loader_stats()
    print(f'DL Models Available: {loader_stats[\"available_models\"]}')
    print(f'DL Models Loaded: {loader_stats[\"loaded_models\"]}')
"
```

### Performance Benchmark
```bash
# Run performance benchmark
python3 -c "
from src.recognition.deep_learning import MTCNNDetector
import time
import numpy as np

detector = MTCNNDetector(device='cpu')

# Test detection speed
sizes = [(224, 224), (480, 640), (720, 1280)]
for size in sizes:
    test_image = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    
    start = time.time()
    faces = detector.detect_faces(test_image)
    duration = time.time() - start
    
    print(f'{size[1]}x{size[0]}: {duration*1000:.1f}ms ({len(faces)} faces)')
"
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Check if modules can be imported
python3 -c "
try:
    import cv2
    print('✅ OpenCV available')
except ImportError:
    print('❌ OpenCV missing: pip3 install opencv-contrib-python')

try:
    import torch
    print('✅ PyTorch available')
except ImportError:
    print('❌ PyTorch missing: pip3 install torch')

try:
    from src.recognition.deep_learning import MTCNNDetector
    print('✅ Deep learning modules available')
except ImportError as e:
    print(f'❌ Deep learning import failed: {e}')
"
```

**2. Camera Issues**
```bash
# Test camera access
python3 -c "
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera accessible')
    cap.release()
else:
    print('❌ Camera not accessible')
    print('Solutions:')
    print('  - Check camera connection')
    print('  - Grant camera permissions')
    print('  - Close other apps using camera')
"
```

**3. Memory Issues**
```bash
# Check memory usage
python3 -c "
import psutil

memory = psutil.virtual_memory()
print(f'Total Memory: {memory.total / 1024**3:.1f} GB')
print(f'Available Memory: {memory.available / 1024**3:.1f} GB')
print(f'Memory Usage: {memory.percent}%')

if memory.available < 2 * 1024**3:
    print('⚠️  Low memory - consider closing other applications')
else:
    print('✅ Sufficient memory available')
"
```

## 📁 Project Structure

```
FaceAttend/
├── main.py                 # Main application entry point
├── quick_test.py           # Quick functionality test
├── test_insightface.py     # InsightFace engine test
├── validate_implementation.py  # Comprehensive validation
├── visual_validation.py    # Visual output generation
├── requirements.txt        # Classical dependencies
├── requirements_dl.txt     # Deep learning dependencies
├── config/                 # Configuration files
├── faces/                  # User face data storage
├── src/                    # Source code
│   ├── recognition/        # Recognition modules
│   │   ├── classical/      # Classical computer vision
│   │   ├── deep_learning/  # Deep learning models
│   │   └── core/           # Hybrid recognition
│   ├── ui/                 # User interface
│   ├── storage/            # Data storage
│   └── utils/              # Utilities
├── models/                 # Deep learning models
└── tests/                  # Test files
```

## 🎯 Next Steps

1. **Start with Classical System**: Run `python3 main.py` to use the original system
2. **Test Deep Learning**: Run `python3 quick_test.py` to verify DL components
3. **Explore Features**: Try face detection, embedding generation, and storage
4. **Check Documentation**: Review `VALIDATION_GUIDE.md` for detailed testing

## 💡 Tips

- **First Time**: Start with classical system to ensure basic functionality
- **Development**: Use validation scripts to test individual components
- **Performance**: Deep learning features work best with good lighting
- **Storage**: Face data is stored in `faces/` directory with metadata

**🎉 You're ready to run FaceAttend with advanced deep learning capabilities!**