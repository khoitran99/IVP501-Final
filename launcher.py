#!/usr/bin/env python3
"""
FaceAttend Launcher - Easy access to all project features
Provides a simple menu to run different components of the system.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print application header."""
    print("=" * 80)
    print("🎯 FaceAttend Project Launcher")
    print("Face Recognition Attendance System with Deep Learning")
    print("=" * 80)

def print_menu():
    """Print main menu options."""
    print("\n📋 Available Options:")
    print()
    print("🏠 MAIN APPLICATION")
    print("  1. Run FaceAttend GUI Application (Classical System)")
    print()
    print("🧠 DEEP LEARNING TESTING")
    print("  2. Quick Test (2 mins) - Basic functionality check")
    print("  3. InsightFace Test - Face recognition engine")
    print("  4. Full Validation - Comprehensive system test")
    print("  5. Visual Validation - Generate detection images")
    print()
    print("🔧 SYSTEM TOOLS")
    print("  6. Check System Status")
    print("  7. Test Camera")
    print("  8. Performance Benchmark")
    print("  9. Install Dependencies")
    print()
    print("📚 DOCUMENTATION")
    print("  10. View Documentation")
    print("  11. Show Project Structure")
    print()
    print("  0. Exit")
    print()

def run_script(script_name, description):
    """Run a Python script with error handling."""
    print(f"\n🚀 {description}")
    print("-" * 60)
    
    try:
        # Get the correct Python executable
        python_exe = sys.executable
        
        # Run the script
        result = subprocess.run([python_exe, script_name], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully")
        else:
            print(f"\n⚠️  {description} completed with warnings")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  {description} interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
    
    input("\nPress Enter to continue...")

def check_system_status():
    """Check and display system status."""
    print("\n🔍 System Status Check")
    print("-" * 60)
    
    try:
        # Check Python version
        print(f"Python Version: {sys.version}")
        
        # Check project directory
        project_dir = Path(__file__).parent
        print(f"Project Directory: {project_dir}")
        
        # Check key files
        key_files = [
            'main.py',
            'requirements.txt', 
            'requirements_dl.txt',
            'quick_test.py',
            'config/models.yaml'
        ]
        
        print("\n📁 Key Files:")
        for file in key_files:
            file_path = project_dir / file
            status = "✅" if file_path.exists() else "❌"
            print(f"  {status} {file}")
        
        # Check dependencies
        print("\n📦 Dependencies:")
        dependencies = [
            ('opencv-python', 'cv2'),
            ('numpy', 'numpy'),
            ('torch', 'torch'),
            ('insightface', 'insightface')
        ]
        
        for pkg_name, import_name in dependencies:
            try:
                __import__(import_name)
                print(f"  ✅ {pkg_name}")
            except ImportError:
                print(f"  ❌ {pkg_name} (not installed)")
        
        # Check deep learning availability
        sys.path.insert(0, str(project_dir))
        try:
            from src.recognition import DL_AVAILABLE
            print(f"\n🧠 Deep Learning: {'✅ Available' if DL_AVAILABLE else '❌ Not Available'}")
        except ImportError:
            print(f"\n🧠 Deep Learning: ❌ Import Error")
        
        # Check user data
        faces_dir = project_dir / 'faces'
        if faces_dir.exists():
            users = [d.name for d in faces_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            print(f"\n👥 Registered Users: {len(users)}")
            if users:
                print(f"  Users: {', '.join(users[:5])}")
                if len(users) > 5:
                    print(f"  ... and {len(users) - 5} more")
        else:
            print(f"\n👥 Registered Users: 0 (faces directory not found)")
            
    except Exception as e:
        print(f"❌ Error checking system status: {e}")
    
    input("\nPress Enter to continue...")

def test_camera():
    """Test camera functionality."""
    print("\n📷 Camera Test")
    print("-" * 60)
    
    try:
        import cv2
        
        print("Testing camera access...")
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Camera is accessible")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera frame captured: {frame.shape}")
                print("✅ Camera is working properly")
            else:
                print("⚠️  Camera opened but cannot read frames")
            
            cap.release()
        else:
            print("❌ Cannot access camera")
            print("\nTroubleshooting:")
            print("  - Check camera connection")
            print("  - Grant camera permissions")
            print("  - Close other apps using camera")
            
    except ImportError:
        print("❌ OpenCV not installed")
        print("Run: pip3 install opencv-contrib-python")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
    
    input("\nPress Enter to continue...")

def run_benchmark():
    """Run performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("-" * 60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        import time
        import numpy as np
        from src.recognition.deep_learning import MTCNNDetector
        
        print("Initializing MTCNN detector...")
        detector = MTCNNDetector(device='cpu')
        print("✅ Detector initialized")
        
        # Test different image sizes
        sizes = [(224, 224), (480, 640), (720, 1280)]
        
        print("\n🔍 Face Detection Speed Test:")
        for size in sizes:
            test_image = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
            
            # Warmup
            detector.detect_faces(test_image)
            
            # Benchmark
            times = []
            for _ in range(3):
                start = time.time()
                faces = detector.detect_faces(test_image)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            print(f"  {size[1]}x{size[0]}: {avg_time:.1f}ms avg")
        
        print("\n✅ Benchmark completed")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("Make sure deep learning dependencies are installed")
    
    input("\nPress Enter to continue...")

def install_dependencies():
    """Install project dependencies."""
    print("\n📦 Installing Dependencies")
    print("-" * 60)
    
    requirements_files = [
        ('requirements.txt', 'Classical Computer Vision'),
        ('requirements_dl.txt', 'Deep Learning')
    ]
    
    for req_file, description in requirements_files:
        if Path(req_file).exists():
            print(f"\n🔄 Installing {description} dependencies...")
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_file],
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {description} dependencies installed")
                else:
                    print(f"⚠️  {description} installation had warnings:")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Failed to install {description} dependencies: {e}")
        else:
            print(f"❌ {req_file} not found")
    
    input("\nPress Enter to continue...")

def show_documentation():
    """Show available documentation."""
    print("\n📚 Available Documentation")
    print("-" * 60)
    
    docs = [
        ('HOW_TO_RUN.md', 'Complete guide on how to run the project'),
        ('VALIDATION_GUIDE.md', 'Validation and testing instructions'),
        ('DEEP_LEARNING_IMPLEMENTATION_PLAN.md', 'Deep learning implementation plan'),
        ('README.md', 'Project overview and setup'),
        ('TECHNICAL_DOCUMENTATION.md', 'Technical details')
    ]
    
    project_dir = Path(__file__).parent
    
    for doc_file, description in docs:
        doc_path = project_dir / doc_file
        status = "✅" if doc_path.exists() else "❌"
        print(f"  {status} {doc_file} - {description}")
    
    print(f"\n💡 To read documentation:")
    print(f"  cat HOW_TO_RUN.md")
    print(f"  open HOW_TO_RUN.md")
    print(f"  code HOW_TO_RUN.md")
    
    input("\nPress Enter to continue...")

def show_project_structure():
    """Show project directory structure."""
    print("\n📁 Project Structure")
    print("-" * 60)
    
    structure = """
FaceAttend/
├── 🚀 launcher.py              # This launcher script
├── 🏠 main.py                  # Main GUI application
├── ⚡ quick_test.py             # Quick functionality test
├── 🧠 test_insightface.py      # InsightFace engine test
├── 🔍 validate_implementation.py # Full validation suite
├── 🎨 visual_validation.py     # Visual output generation
├── 📋 requirements.txt         # Classical dependencies  
├── 🧠 requirements_dl.txt      # Deep learning dependencies
├── ⚙️  config/                 # Configuration files
├── 👥 faces/                   # User face data storage
├── 📊 logs/                    # Application logs
├── 🎯 models/                  # Deep learning models
├── 🔧 src/                     # Source code
│   ├── 🎯 recognition/         # Recognition modules
│   │   ├── 📷 classical/       # Classical computer vision
│   │   ├── 🧠 deep_learning/   # Deep learning models
│   │   └── ⚙️  core/           # Hybrid recognition
│   ├── 🎨 ui/                  # User interface
│   ├── 💾 storage/             # Data storage
│   └── 🛠️  utils/              # Utilities
└── 🧪 tests/                   # Test files
    """
    
    print(structure)
    input("\nPress Enter to continue...")

def main():
    """Main launcher loop."""
    while True:
        try:
            print_header()
            print_menu()
            
            choice = input("👉 Enter your choice (0-11): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye!")
                break
                
            elif choice == '1':
                run_script('main.py', 'FaceAttend GUI Application')
                
            elif choice == '2':
                run_script('quick_test.py', 'Quick Functionality Test')
                
            elif choice == '3':
                run_script('test_insightface.py', 'InsightFace Recognition Engine Test')
                
            elif choice == '4':
                run_script('validate_implementation.py', 'Full System Validation')
                
            elif choice == '5':
                run_script('visual_validation.py', 'Visual Validation with Output Images')
                
            elif choice == '6':
                check_system_status()
                
            elif choice == '7':
                test_camera()
                
            elif choice == '8':
                run_benchmark()
                
            elif choice == '9':
                install_dependencies()
                
            elif choice == '10':
                show_documentation()
                
            elif choice == '11':
                show_project_structure()
                
            else:
                print(f"\n❌ Invalid choice: {choice}")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()