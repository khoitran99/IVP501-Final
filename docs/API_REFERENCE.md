# API Reference & Usage Guide

## 📖 Overview

This document provides comprehensive API reference and usage examples for the FaceAttend deep learning face recognition system.

## 📋 Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Core API Classes](#core-api-classes)
3. [Recognition Operations](#recognition-operations)
4. [Storage Operations](#storage-operations)
5. [Configuration API](#configuration-api)
6. [UI Integration](#ui-integration)
7. [Error Handling](#error-handling)
8. [Performance Tuning](#performance-tuning)
9. [Examples & Tutorials](#examples--tutorials)

## 🚀 Quick Start Guide

### Basic Usage Example

```python
import cv2
from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer
from src.storage.face_storage import FaceStorage

# Initialize the system
recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.6)
storage = FaceStorage()

# Register a new user
user_id = "john_001"
face_images = []  # Load your face images here
success = recognizer.train_user(user_id, face_images)

# Recognize a face
test_image = cv2.imread("test_face.jpg")
user_id, similarity = recognizer.recognize_face(test_image)
print(f"Recognized: {user_id} with similarity: {similarity}")
```

### Real-time Recognition Example

```python
from src.recognition.realtime_recognizer import RealtimeRecognizer

# Initialize real-time recognizer
recognizer = RealtimeRecognizer(similarity_threshold=0.6)

# Set up callbacks
def on_recognition(result):
    print(f"User: {result['name']}, Similarity: {result['confidence']}")

recognizer.set_callbacks(recognition_callback=on_recognition)

# Start recognition
recognizer.start_recognition()
# Recognition runs in background thread
# Call recognizer.stop_recognition() when done
```

## 🔧 Core API Classes

### SimpleInsightFaceRecognizer

Primary deep learning face recognition engine.

#### Constructor

```python
SimpleInsightFaceRecognizer(confidence_threshold: float = 0.6)
```

**Parameters:**
- `confidence_threshold` (float): Similarity threshold for recognition (0.0-1.0)

**Example:**
```python
recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.7)
```

#### Methods

##### `generate_embedding(face_image: np.ndarray) -> Optional[np.ndarray]`

Generate 512-dimensional face embedding from image.

**Parameters:**
- `face_image` (np.ndarray): Face image as BGR numpy array

**Returns:**
- `np.ndarray`: 512D embedding vector, or `None` if no face detected

**Example:**
```python
import cv2
face_image = cv2.imread("face.jpg")
embedding = recognizer.generate_embedding(face_image)
if embedding is not None:
    print(f"Generated embedding with shape: {embedding.shape}")
```

##### `train_user(user_id: str, face_images: List[np.ndarray]) -> bool`

Train the system with multiple face images for a user.

**Parameters:**
- `user_id` (str): Unique identifier for the user
- `face_images` (List[np.ndarray]): List of face images as numpy arrays

**Returns:**
- `bool`: `True` if training successful, `False` otherwise

**Example:**
```python
# Load multiple face images for better accuracy
face_images = []
for i in range(1, 6):
    img = cv2.imread(f"user_faces/john_{i}.jpg")
    face_images.append(img)

success = recognizer.train_user("john_001", face_images)
if success:
    print("User trained successfully")
```

##### `recognize_face(face_image: np.ndarray) -> Tuple[Optional[str], float]`

Recognize a face from an image.

**Parameters:**
- `face_image` (np.ndarray): Face image as numpy array

**Returns:**
- `Tuple[Optional[str], float]`: (user_id, similarity_score)
  - `user_id`: Recognized user ID or `None` if not recognized
  - `similarity_score`: Similarity confidence (0.0-1.0)

**Example:**
```python
test_image = cv2.imread("unknown_face.jpg")
user_id, similarity = recognizer.recognize_face(test_image)

if user_id:
    print(f"Recognized as: {user_id} (similarity: {similarity:.3f})")
else:
    print(f"Unknown person (similarity: {similarity:.3f})")
```

##### `get_model_info() -> Dict[str, Any]`

Get information about the recognition model.

**Returns:**
- `Dict`: Model information including users, embeddings, and status

**Example:**
```python
info = recognizer.get_model_info()
print(f"Total users: {info['total_users']}")
print(f"Total embeddings: {info['total_embeddings']}")
print(f"Model trained: {info['is_trained']}")
```

##### `set_confidence_threshold(threshold: float) -> None`

Update the recognition confidence threshold.

**Parameters:**
- `threshold` (float): New threshold value (0.0-1.0)

**Example:**
```python
# Make recognition more strict
recognizer.set_confidence_threshold(0.8)
```

##### `train_model() -> Dict[str, Any]`

Train the model with all registered users.

**Returns:**
- `Dict`: Training result with success status and statistics

**Example:**
```python
result = recognizer.train_model()
if result['success']:
    print(f"Trained {result['users_count']} users successfully")
else:
    print(f"Training failed: {result.get('error', 'Unknown error')}")
```

### RealtimeRecognizer

High-level interface for real-time face recognition.

#### Constructor

```python
RealtimeRecognizer(
    camera_manager: CameraManager = None,
    similarity_threshold: float = 0.6,
    recognition_interval: float = 1.0
)
```

**Parameters:**
- `camera_manager` (CameraManager): Custom camera manager instance
- `similarity_threshold` (float): Recognition similarity threshold
- `recognition_interval` (float): Minimum time between recognition attempts

#### Methods

##### `set_callbacks(status_callback, recognition_callback, frame_callback)`

Set callback functions for real-time updates.

**Parameters:**
- `status_callback` (Callable[[str], None]): Called with status updates
- `recognition_callback` (Callable[[Dict], None]): Called with recognition results
- `frame_callback` (Callable[[np.ndarray], None]): Called with processed frames

**Example:**
```python
def on_status_update(status: str):
    print(f"Status: {status}")

def on_recognition_result(result: Dict):
    if result['user_id']:
        print(f"Recognized: {result['name']} ({result['confidence']:.3f})")
    else:
        print("Unknown person detected")

def on_frame_update(frame: np.ndarray):
    # Display frame or process it
    cv2.imshow("Recognition", frame)

recognizer.set_callbacks(
    status_callback=on_status_update,
    recognition_callback=on_recognition_result,
    frame_callback=on_frame_update
)
```

##### `start_recognition() -> bool`

Start real-time recognition process.

**Returns:**
- `bool`: `True` if started successfully

**Example:**
```python
if recognizer.start_recognition():
    print("Real-time recognition started")
    # Recognition runs in background thread
else:
    print("Failed to start recognition")
```

##### `stop_recognition() -> None`

Stop real-time recognition process.

**Example:**
```python
recognizer.stop_recognition()
print("Recognition stopped")
```

##### `retrain_model() -> bool`

Retrain the recognition model with current data.

**Returns:**
- `bool`: `True` if retraining successful

**Example:**
```python
if recognizer.retrain_model():
    print("Model retrained successfully")
else:
    print("Model retraining failed")
```

##### `update_similarity_threshold(threshold: float) -> None`

Update similarity threshold during runtime.

**Parameters:**
- `threshold` (float): New threshold (0.0-1.0)

**Example:**
```python
# Make recognition more lenient
recognizer.update_similarity_threshold(0.4)
```

##### `get_recognition_stats() -> Dict`

Get comprehensive recognition statistics.

**Returns:**
- `Dict`: Statistics including model info, system status, and attendance data

**Example:**
```python
stats = recognizer.get_recognition_stats()
print(f"Model info: {stats['model_info']}")
print(f"System status: {stats['system_status']}")
print(f"Is running: {stats['is_running']}")
```

## 💾 Storage Operations

### FaceStorage

Manages face image and user data storage.

#### Constructor

```python
FaceStorage(base_dir: str = "faces")
```

#### Methods

##### `save_user_face(user_id: str, face_image: np.ndarray, metadata: Dict = None) -> str`

Save a face image for a user.

**Parameters:**
- `user_id` (str): User identifier
- `face_image` (np.ndarray): Face image array
- `metadata` (Dict): Optional metadata

**Returns:**
- `str`: Path to saved image

**Example:**
```python
storage = FaceStorage()
face_image = cv2.imread("new_face.jpg")
image_path = storage.save_user_face("john_001", face_image)
print(f"Face saved to: {image_path}")
```

##### `get_user_face_images(user_id: str) -> List[np.ndarray]`

Load all face images for a user.

**Parameters:**
- `user_id` (str): User identifier

**Returns:**
- `List[np.ndarray]`: List of face image arrays

**Example:**
```python
face_images = storage.get_user_face_images("john_001")
print(f"Loaded {len(face_images)} face images for user")
```

##### `register_user(user_info: Dict, face_images: List[np.ndarray]) -> Dict`

Register a new user with complete information.

**Parameters:**
- `user_info` (Dict): User information (name, email, etc.)
- `face_images` (List[np.ndarray]): Face images

**Returns:**
- `Dict`: Registration result

**Example:**
```python
user_info = {
    "name": "John Doe",
    "email": "john@example.com",
    "department": "Engineering"
}

face_images = [cv2.imread(f"faces/john_{i}.jpg") for i in range(1, 6)]

result = storage.register_user(user_info, face_images)
if result['success']:
    print(f"User registered with ID: {result['user_id']}")
```

##### `list_users() -> List[str]`

Get list of all registered user IDs.

**Returns:**
- `List[str]`: List of user IDs

**Example:**
```python
users = storage.list_users()
print(f"Registered users: {users}")
```

##### `get_user_info(user_id: str) -> Dict`

Get user information by ID.

**Parameters:**
- `user_id` (str): User identifier

**Returns:**
- `Dict`: User information

**Example:**
```python
user_info = storage.get_user_info("john_001")
print(f"User name: {user_info['name']}")
print(f"Department: {user_info['department']}")
```

##### `delete_user(user_id: str) -> bool`

Delete a user and all associated data.

**Parameters:**
- `user_id` (str): User identifier

**Returns:**
- `bool`: `True` if deletion successful

**Example:**
```python
if storage.delete_user("john_001"):
    print("User deleted successfully")
```

### AttendanceLogger

Manages attendance logging and data retrieval.

#### Methods

##### `log_attendance(user_id: str, name: str, confidence: float) -> bool`

Log an attendance entry.

**Parameters:**
- `user_id` (str): User identifier
- `name` (str): User name
- `confidence` (float): Recognition confidence score

**Returns:**
- `bool`: `True` if logged successfully (not duplicate)

**Example:**
```python
logger = AttendanceLogger()
logged = logger.log_attendance("john_001", "John Doe", 0.875)
if logged:
    print("Attendance logged")
else:
    print("Duplicate entry - not logged")
```

##### `get_daily_attendance(date: str = None) -> List[Dict]`

Get attendance records for a specific date.

**Parameters:**
- `date` (str): Date in YYYY-MM-DD format (None for today)

**Returns:**
- `List[Dict]`: List of attendance records

**Example:**
```python
# Get today's attendance
today_records = logger.get_daily_attendance()

# Get specific date attendance
records = logger.get_daily_attendance("2024-01-15")
for record in records:
    print(f"{record['time']} - {record['name']} ({record['confidence']:.3f})")
```

##### `get_date_range_attendance(start_date: str, end_date: str) -> List[Dict]`

Get attendance records for a date range.

**Parameters:**
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)

**Returns:**
- `List[Dict]`: List of attendance records

**Example:**
```python
records = logger.get_date_range_attendance("2024-01-01", "2024-01-31")
print(f"Found {len(records)} attendance records in January")
```

##### `export_to_csv(start_date: str, end_date: str, filename: str = None) -> str`

Export attendance data to CSV file.

**Parameters:**
- `start_date` (str): Start date
- `end_date` (str): End date
- `filename` (str): Output filename (optional)

**Returns:**
- `str`: Path to exported CSV file

**Example:**
```python
csv_path = logger.export_to_csv("2024-01-01", "2024-01-31", "january_attendance.csv")
print(f"Attendance exported to: {csv_path}")
```

##### `get_attendance_statistics(date_range: Tuple[str, str] = None) -> Dict`

Get attendance statistics.

**Parameters:**
- `date_range` (Tuple[str, str]): Optional date range

**Returns:**
- `Dict`: Statistics including totals, averages, etc.

**Example:**
```python
stats = logger.get_attendance_statistics()
print(f"Total entries: {stats['total_entries']}")
print(f"Unique users: {stats['unique_users']}")
print(f"Average confidence: {stats['average_confidence']:.3f}")
```

## ⚙️ Configuration API

### ConfigurationManager

Manages system configuration.

#### Example Usage

```python
from src.utils.config_manager import ConfigurationManager

config = ConfigurationManager()

# Get current settings
current_threshold = config.get('recognition_settings.similarity_threshold')
print(f"Current threshold: {current_threshold}")

# Update settings
config.update({
    'recognition_settings.similarity_threshold': 0.7,
    'preprocessing_settings.face_alignment': True
})

# Save changes
config.save()
```

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `recognition_settings.similarity_threshold` | float | 0.6 | Recognition threshold |
| `recognition_settings.method` | str | "deep_learning" | Recognition method |
| `preprocessing_settings.face_alignment` | bool | true | Enable face alignment |
| `preprocessing_settings.quality_threshold` | float | 0.7 | Image quality threshold |
| `storage_settings.save_embeddings` | bool | true | Save embeddings to disk |
| `performance_settings.max_recognition_fps` | int | 15 | Maximum FPS for recognition |

## 🖥️ UI Integration

### Camera Widget Integration

```python
from src.ui.camera_widget import CameraWidget
import tkinter as tk

class MyApp:
    def __init__(self):
        self.root = tk.Tk()
        self.camera_widget = CameraWidget(self.root, width=640, height=480)
        self.camera_widget.pack()
        
        # Set callback for frame processing
        self.camera_widget.set_frame_callback(self.process_frame)
    
    def process_frame(self, frame):
        # Process frame for recognition
        # Return modified frame for display
        return frame
```

### Custom UI Components

```python
import tkinter as tk
from tkinter import ttk
from src.recognition.realtime_recognizer import RealtimeRecognizer

class CustomRecognitionUI:
    def __init__(self):
        self.root = tk.Tk()
        self.recognizer = RealtimeRecognizer()
        self.setup_ui()
    
    def setup_ui(self):
        # Threshold control
        self.threshold_var = tk.DoubleVar(value=0.6)
        threshold_scale = ttk.Scale(
            self.root,
            from_=0.3, to=0.9,
            variable=self.threshold_var,
            command=self.on_threshold_change
        )
        threshold_scale.pack()
        
        # Control buttons
        start_btn = ttk.Button(self.root, text="Start", command=self.start_recognition)
        stop_btn = ttk.Button(self.root, text="Stop", command=self.stop_recognition)
        start_btn.pack()
        stop_btn.pack()
    
    def on_threshold_change(self, value):
        threshold = float(value)
        self.recognizer.update_similarity_threshold(threshold)
    
    def start_recognition(self):
        self.recognizer.set_callbacks(
            recognition_callback=self.on_recognition
        )
        self.recognizer.start_recognition()
    
    def stop_recognition(self):
        self.recognizer.stop_recognition()
    
    def on_recognition(self, result):
        print(f"Recognition result: {result}")
```

## ❌ Error Handling

### Exception Types

```python
from src.utils.exceptions import (
    FaceRecognitionError,
    CameraError,
    StorageError,
    ConfigurationError
)

# Handle recognition errors
try:
    result = recognizer.recognize_face(image)
except FaceRecognitionError as e:
    print(f"Recognition error: {e}")
    # Handle error appropriately

# Handle camera errors
try:
    recognizer.start_recognition()
except CameraError as e:
    print(f"Camera error: {e}")
    # Inform user about camera issues

# Handle storage errors
try:
    storage.save_user_face(user_id, image)
except StorageError as e:
    print(f"Storage error: {e}")
    # Handle storage issues
```

### Error Recovery Strategies

```python
def robust_recognition(recognizer, image, max_retries=3):
    """Robust recognition with retry logic"""
    for attempt in range(max_retries):
        try:
            return recognizer.recognize_face(image)
        except FaceRecognitionError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Recognition attempt {attempt + 1} failed, retrying...")
            time.sleep(0.1)
    
    return None, 0.0

def safe_model_training(recognizer):
    """Safe model training with fallback"""
    try:
        result = recognizer.train_model()
        if result['success']:
            return True
    except Exception as e:
        print(f"Training failed: {e}")
    
    # Fallback: Try with lower quality threshold
    try:
        # Implement fallback training strategy
        return recognizer.train_with_fallback()
    except Exception:
        return False
```

## 🚀 Performance Tuning

### Optimization Strategies

#### Batch Processing

```python
def batch_recognition(recognizer, images):
    """Process multiple images efficiently"""
    results = []
    
    # Generate embeddings in batch
    embeddings = []
    for image in images:
        embedding = recognizer.generate_embedding(image)
        if embedding is not None:
            embeddings.append(embedding)
    
    # Batch similarity calculation
    if embeddings:
        # Vectorized similarity computation
        similarities = compute_batch_similarities(embeddings)
        results = process_batch_results(similarities)
    
    return results
```

#### Memory Management

```python
import gc

def memory_efficient_training(recognizer, user_data):
    """Memory-efficient training for large datasets"""
    for user_id, images in user_data.items():
        # Process user data in chunks
        chunk_size = 10
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            recognizer.train_user_chunk(user_id, chunk)
            
            # Force garbage collection
            gc.collect()
    
    return True
```

#### Caching Strategies

```python
from functools import lru_cache

class CachedRecognizer:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.embedding_cache = {}
    
    @lru_cache(maxsize=1000)
    def cached_embedding_generation(self, image_hash):
        """Cache embeddings by image hash"""
        return self.recognizer.generate_embedding(image)
    
    def smart_recognition(self, image):
        """Recognition with intelligent caching"""
        image_hash = hash(image.tobytes())
        
        if image_hash in self.embedding_cache:
            embedding = self.embedding_cache[image_hash]
        else:
            embedding = self.recognizer.generate_embedding(image)
            self.embedding_cache[image_hash] = embedding
        
        return self.recognizer.match_embedding(embedding)
```

## 📚 Examples & Tutorials

### Tutorial 1: Basic Face Registration

```python
import cv2
from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer
from src.storage.face_storage import FaceStorage

def register_new_user():
    # Initialize components
    recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.6)
    storage = FaceStorage()
    
    # Prepare user information
    user_info = {
        "name": "Alice Johnson",
        "email": "alice@company.com",
        "department": "Marketing",
        "employee_id": "EMP001"
    }
    
    # Load face images (multiple angles recommended)
    face_images = []
    image_paths = [
        "alice_front.jpg",
        "alice_left.jpg", 
        "alice_right.jpg",
        "alice_smile.jpg",
        "alice_neutral.jpg"
    ]
    
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            face_images.append(image)
    
    # Register user
    result = storage.register_user(user_info, face_images)
    
    if result['success']:
        user_id = result['user_id']
        print(f"✅ User registered successfully!")
        print(f"   User ID: {user_id}")
        print(f"   Images processed: {result['images_processed']}")
        
        # Train the recognition model
        if recognizer.train_user(user_id, face_images):
            print("✅ User trained in recognition model")
        else:
            print("❌ Training failed")
    else:
        print(f"❌ Registration failed: {result['error']}")

if __name__ == "__main__":
    register_new_user()
```

### Tutorial 2: Real-time Attendance System

```python
import cv2
import tkinter as tk
from tkinter import ttk
from src.recognition.realtime_recognizer import RealtimeRecognizer
from src.storage.attendance_logger import AttendanceLogger

class AttendanceSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-time Attendance System")
        
        # Initialize recognition components
        self.recognizer = RealtimeRecognizer(similarity_threshold=0.6)
        self.logger = AttendanceLogger()
        
        # Setup UI
        self.setup_ui()
        
        # Configure callbacks
        self.recognizer.set_callbacks(
            status_callback=self.update_status,
            recognition_callback=self.handle_recognition
        )
    
    def setup_ui(self):
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(pady=10)
        
        # Recognition info
        self.info_var = tk.StringVar(value="No recognition yet")
        info_label = ttk.Label(self.root, textvariable=self.info_var)
        info_label.pack(pady=10)
        
        # Threshold control
        ttk.Label(self.root, text="Similarity Threshold:").pack()
        self.threshold_var = tk.DoubleVar(value=0.6)
        threshold_scale = ttk.Scale(
            self.root,
            from_=0.3, to=0.8,
            variable=self.threshold_var,
            command=self.update_threshold
        )
        threshold_scale.pack(pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Attendance",
            command=self.start_attendance
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop Attendance",
            command=self.stop_attendance,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Today's attendance list
        ttk.Label(self.root, text="Today's Attendance:").pack(pady=(20, 5))
        
        self.attendance_listbox = tk.Listbox(self.root, height=8)
        self.attendance_listbox.pack(pady=5, padx=20, fill=tk.X)
        
        # Refresh attendance list
        self.refresh_attendance()
    
    def update_threshold(self, value):
        threshold = float(value)
        self.recognizer.update_similarity_threshold(threshold)
        print(f"Updated threshold to: {threshold:.2f}")
    
    def start_attendance(self):
        if self.recognizer.start_recognition():
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            print("✅ Attendance tracking started")
        else:
            print("❌ Failed to start attendance tracking")
    
    def stop_attendance(self):
        self.recognizer.stop_recognition()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        print("⏹️ Attendance tracking stopped")
    
    def update_status(self, status):
        self.status_var.set(f"Status: {status}")
    
    def handle_recognition(self, result):
        if result['user_id']:
            # Successful recognition
            user_name = result['name']
            confidence = result['confidence']
            
            self.info_var.set(
                f"✅ {user_name} recognized (similarity: {confidence:.3f})"
            )
            
            # Log attendance
            attendance_logged = self.logger.log_attendance(
                result['user_id'],
                user_name,
                confidence
            )
            
            if attendance_logged:
                print(f"📝 Attendance logged for {user_name}")
                self.refresh_attendance()
            else:
                print(f"⚠️ Duplicate entry for {user_name}")
        
        else:
            # Unknown person
            confidence = result.get('confidence', 0.0)
            self.info_var.set(f"❌ Unknown person (similarity: {confidence:.3f})")
    
    def refresh_attendance(self):
        # Clear current list
        self.attendance_listbox.delete(0, tk.END)
        
        # Get today's attendance
        today_records = self.logger.get_daily_attendance()
        
        if not today_records:
            self.attendance_listbox.insert(0, "No attendance records for today")
            return
        
        # Add records to list
        for record in today_records[-10:]:  # Show last 10 records
            time_str = record['time']
            name = record['name']
            confidence = record['confidence']
            
            entry = f"{time_str} - {name} ({confidence:.3f})"
            self.attendance_listbox.insert(tk.END, entry)
    
    def run(self):
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self.recognizer.stop_recognition()

if __name__ == "__main__":
    app = AttendanceSystem()
    app.run()
```

### Tutorial 3: Batch Processing System

```python
import os
import cv2
from pathlib import Path
from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer

class BatchProcessor:
    def __init__(self):
        self.recognizer = SimpleInsightFaceRecognizer(confidence_threshold=0.6)
    
    def process_directory(self, image_directory, output_file="recognition_results.txt"):
        """Process all images in a directory"""
        image_directory = Path(image_directory)
        results = []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_directory.glob(f"*{ext}"))
            image_files.extend(image_directory.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  ❌ Could not load image")
                continue
            
            # Recognize face
            user_id, similarity = self.recognizer.recognize_face(image)
            
            result = {
                'image_path': str(image_path),
                'user_id': user_id,
                'similarity': similarity,
                'recognized': user_id is not None
            }
            
            results.append(result)
            
            # Print result
            if user_id:
                print(f"  ✅ Recognized as: {user_id} (similarity: {similarity:.3f})")
            else:
                print(f"  ❌ Unknown person (similarity: {similarity:.3f})")
        
        # Save results
        self.save_results(results, output_file)
        self.print_summary(results)
        
        return results
    
    def save_results(self, results, output_file):
        """Save results to text file"""
        with open(output_file, 'w') as f:
            f.write("Batch Recognition Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Image: {result['image_path']}\n")
                f.write(f"User ID: {result['user_id']}\n")
                f.write(f"Similarity: {result['similarity']:.4f}\n")
                f.write(f"Recognized: {result['recognized']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"📄 Results saved to: {output_file}")
    
    def print_summary(self, results):
        """Print processing summary"""
        total_images = len(results)
        recognized_count = sum(1 for r in results if r['recognized'])
        unknown_count = total_images - recognized_count
        
        if total_images > 0:
            recognition_rate = (recognized_count / total_images) * 100
            avg_similarity_recognized = sum(
                r['similarity'] for r in results if r['recognized']
            ) / max(recognized_count, 1)
            
            avg_similarity_unknown = sum(
                r['similarity'] for r in results if not r['recognized']
            ) / max(unknown_count, 1)
        else:
            recognition_rate = 0
            avg_similarity_recognized = 0
            avg_similarity_unknown = 0
        
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {total_images}")
        print(f"Recognized faces: {recognized_count}")
        print(f"Unknown faces: {unknown_count}")
        print(f"Recognition rate: {recognition_rate:.1f}%")
        print(f"Avg similarity (recognized): {avg_similarity_recognized:.3f}")
        print(f"Avg similarity (unknown): {avg_similarity_unknown:.3f}")

def main():
    processor = BatchProcessor()
    
    # Process directory of images
    image_directory = "test_images/"  # Change this to your directory
    
    if os.path.exists(image_directory):
        results = processor.process_directory(image_directory)
    else:
        print(f"❌ Directory not found: {image_directory}")

if __name__ == "__main__":
    main()
```

### Tutorial 4: Custom Similarity Metrics

```python
import numpy as np
from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer

class AdvancedRecognizer(SimpleInsightFaceRecognizer):
    """Extended recognizer with custom similarity metrics"""
    
    def __init__(self, confidence_threshold=0.6, similarity_metric='cosine'):
        super().__init__(confidence_threshold)
        self.similarity_metric = similarity_metric
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate similarity using specified metric"""
        if self.similarity_metric == 'cosine':
            return self.cosine_similarity(embedding1, embedding2)
        elif self.similarity_metric == 'euclidean':
            return self.euclidean_similarity(embedding1, embedding2)
        elif self.similarity_metric == 'manhattan':
            return self.manhattan_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def cosine_similarity(self, a, b):
        """Cosine similarity (default)"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def euclidean_similarity(self, a, b):
        """Euclidean distance converted to similarity"""
        distance = np.linalg.norm(a - b)
        # Convert to similarity (higher = more similar)
        return 1.0 / (1.0 + distance)
    
    def manhattan_similarity(self, a, b):
        """Manhattan distance converted to similarity"""
        distance = np.sum(np.abs(a - b))
        # Convert to similarity
        return 1.0 / (1.0 + distance)
    
    def recognize_face_advanced(self, face_image, return_all_scores=False):
        """Advanced recognition with multiple metrics"""
        query_embedding = self.generate_embedding(face_image)
        if query_embedding is None:
            return None, 0.0, {}
        
        if not self.user_embeddings:
            return None, 0.0, {}
        
        all_scores = {}
        best_user_id = None
        best_similarity = 0.0
        
        for user_id, embeddings in self.user_embeddings.items():
            user_scores = []
            
            for embedding in embeddings:
                similarity = self.calculate_similarity(query_embedding, embedding)
                user_scores.append(similarity)
            
            # Use maximum similarity for this user
            max_similarity = max(user_scores) if user_scores else 0.0
            all_scores[user_id] = {
                'max_similarity': max_similarity,
                'avg_similarity': np.mean(user_scores) if user_scores else 0.0,
                'min_similarity': min(user_scores) if user_scores else 0.0,
                'embedding_count': len(user_scores)
            }
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_user_id = user_id
        
        # Check threshold
        if best_similarity >= self.confidence_threshold:
            recognized_user = best_user_id
        else:
            recognized_user = None
        
        if return_all_scores:
            return recognized_user, best_similarity, all_scores
        else:
            return recognized_user, best_similarity

# Usage example
def compare_similarity_metrics():
    """Compare different similarity metrics"""
    import cv2
    
    # Test with different metrics
    metrics = ['cosine', 'euclidean', 'manhattan']
    test_image = cv2.imread("test_face.jpg")
    
    for metric in metrics:
        print(f"\nTesting with {metric} similarity:")
        recognizer = AdvancedRecognizer(
            confidence_threshold=0.6,
            similarity_metric=metric
        )
        
        # Assuming some users are already trained
        user_id, similarity, all_scores = recognizer.recognize_face_advanced(
            test_image, 
            return_all_scores=True
        )
        
        print(f"  Best match: {user_id}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  All scores: {all_scores}")

if __name__ == "__main__":
    compare_similarity_metrics()
```

---

**This API reference provides comprehensive documentation for integrating and extending the FaceAttend deep learning face recognition system.**