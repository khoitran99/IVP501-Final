#!/usr/bin/env python3
"""
FaceAttend - Python Desktop Face Recognition Attendance System
Main application entry point

Author: FaceAttend Team
Version: 1.0
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import os
from datetime import datetime

# Import application modules
try:
    from src.ui.camera_widget import CameraWidget
    from src.camera.camera_manager import test_camera
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all modules are properly installed")

class FaceAttendApp:
    """Main application class for FaceAttend"""
    
    def __init__(self):
        """Initialize the FaceAttend application"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting FaceAttend Application")
        
        # Create main window
        self.root = tk.Tk()
        self.setup_main_window()
        
        # Initialize main window
        self.create_main_window()
    
    def setup_logging(self):
        """Set up basic logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        log_filename = f"logs/faceattend_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_main_window(self):
        """Configure main window properties"""
        self.root.title("FaceAttend - Face Recognition Attendance System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Mac-specific settings
        if sys.platform == "darwin":
            # Set Mac-specific window properties
            self.root.tk.call('tk', 'scaling', 1.0)
            
            # Handle window close event
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_main_window(self):
        """Create main window for Phase 1"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Home tab
        self.create_home_tab()
        
        # Camera test tab
        self.create_camera_tab()
        
        # Status area
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_var = tk.StringVar()
        self.status_var.set("System ready - Phase 1 implementation")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack()
    
    def create_home_tab(self):
        """Create the home tab"""
        home_frame = ttk.Frame(self.notebook)
        self.notebook.add(home_frame, text="Home")
        
        # Configure grid
        home_frame.columnconfigure(0, weight=1)
        
        # Title section
        title_frame = ttk.Frame(home_frame)
        title_frame.pack(pady=20)
        
        title_label = ttk.Label(
            title_frame, 
            text="FaceAttend", 
            font=("Arial", 28, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame, 
            text="Face Recognition Attendance System", 
            font=("Arial", 14)
        )
        subtitle_label.pack(pady=(5, 0))
        
        version_label = ttk.Label(
            title_frame, 
            text="Version 1.0 - Phase 1 Implementation", 
            font=("Arial", 10),
            foreground="gray"
        )
        version_label.pack(pady=(5, 0))
        
        # Features section
        features_frame = ttk.LabelFrame(home_frame, text="Available Features", padding=20)
        features_frame.pack(pady=20, padx=40, fill=tk.X)
        
        # Navigation buttons grid
        button_frame = ttk.Frame(features_frame)
        button_frame.pack()
        
        # Row 1
        register_btn = ttk.Button(
            button_frame,
            text="📷 Register Face",
            command=self.open_register_screen,
            width=25
        )
        register_btn.grid(row=0, column=0, padx=10, pady=10)
        
        attendance_btn = ttk.Button(
            button_frame,
            text="👤 Start Attendance",
            command=self.open_attendance_screen,
            width=25
        )
        attendance_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Row 2
        logs_btn = ttk.Button(
            button_frame,
            text="📊 View Attendance Logs",
            command=self.open_logs_screen,
            width=25
        )
        logs_btn.grid(row=1, column=0, padx=10, pady=10)
        
        camera_btn = ttk.Button(
            button_frame,
            text="🎥 Test Camera",
            command=self.test_camera,
            width=25
        )
        camera_btn.grid(row=1, column=1, padx=10, pady=10)
        
        # Info section
        info_frame = ttk.LabelFrame(home_frame, text="Phase 1 Status", padding=15)
        info_frame.pack(pady=20, padx=40, fill=tk.X)
        
        info_text = """
Phase 1 Implementation Complete ✓
• Basic Tkinter GUI ✓
• Webcam access and video streaming ✓
• Project structure and logging ✓
• Error handling framework ✓

Coming in Future Phases:
• Face registration (Phase 2)
• Face recognition (Phase 3)  
• Attendance logging (Phase 4)
• Performance optimization (Phase 5)
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack()
    
    def create_camera_tab(self):
        """Create the camera test tab"""
        camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(camera_frame, text="Camera Test")
        
        # Instructions
        instructions = ttk.Label(
            camera_frame,
            text="Use this tab to test your camera functionality.\nThis will be used for face registration and attendance capture in later phases.",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        instructions.pack(pady=20)
        
        # Camera widget
        try:
            self.camera_widget = CameraWidget(camera_frame, width=640, height=480)
            self.camera_widget.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        except Exception as e:
            self.logger.error(f"Failed to create camera widget: {str(e)}")
            error_label = ttk.Label(
                camera_frame,
                text=f"Camera widget unavailable: {str(e)}",
                foreground="red"
            )
            error_label.pack(pady=20)
    
    def open_register_screen(self):
        """Open face registration screen (placeholder for Phase 1)"""
        self.status_var.set("Register Face - Feature coming in Phase 2")
        messagebox.showinfo(
            "Feature Coming Soon", 
            "Face registration will be implemented in Phase 2.\n\n"
            "This will include:\n"
            "• Multiple image capture per user\n"
            "• Face detection validation\n"
            "• Image preprocessing\n"
            "• User ID management"
        )
    
    def open_attendance_screen(self):
        """Open attendance capture screen (placeholder for Phase 1)"""
        self.status_var.set("Attendance Capture - Feature coming in Phase 3")
        messagebox.showinfo(
            "Feature Coming Soon", 
            "Attendance capture will be implemented in Phase 3.\n\n"
            "This will include:\n"
            "• Real-time face recognition\n"
            "• Automatic attendance marking\n"
            "• LBPH recognition engine\n"
            "• Confidence threshold management"
        )
    
    def open_logs_screen(self):
        """Open attendance logs screen (placeholder for Phase 1)"""
        self.status_var.set("View Logs - Feature coming in Phase 4")
        messagebox.showinfo(
            "Feature Coming Soon", 
            "Log viewing will be implemented in Phase 4.\n\n"
            "This will include:\n"
            "• Daily CSV log viewing\n"
            "• Attendance statistics\n"
            "• Export functionality\n"
            "• Date range filtering"
        )
    
    def test_camera(self):
        """Test camera functionality (Phase 1 implementation)"""
        self.status_var.set("Testing camera...")
        
        try:
            # Test camera using the camera manager
            success, message = test_camera(0)
            
            if success:
                self.status_var.set("Camera test successful!")
                messagebox.showinfo("Camera Test", f"✓ {message}\n\nYou can also test the live camera feed in the 'Camera Test' tab.")
                self.logger.info("Camera test successful")
            else:
                self.status_var.set("Camera test failed")
                messagebox.showerror("Camera Error", f"✗ {message}\n\nPlease check:\n• Camera is connected\n• Camera permissions are granted\n• No other app is using the camera")
                self.logger.error(f"Camera test failed: {message}")
                
        except Exception as e:
            error_msg = f"Camera test failed: {str(e)}"
            self.status_var.set("Camera test failed")
            messagebox.showerror("Camera Error", error_msg)
            self.logger.error(error_msg)
    
    def on_closing(self):
        """Handle application closing"""
        try:
            self.logger.info("Application closing")
            
            # Clean up camera widget if it exists
            if hasattr(self, 'camera_widget'):
                self.camera_widget.destroy()
            
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error during application shutdown: {str(e)}")
            self.root.destroy()
    
    def run(self):
        """Start the application main loop"""
        try:
            self.logger.info("Application started successfully")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            messagebox.showerror("Application Error", f"An error occurred: {str(e)}")
        finally:
            self.logger.info("Application shutdown")

def main():
    """Main entry point"""
    try:
        app = FaceAttendApp()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 