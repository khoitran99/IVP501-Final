"""
Recognition Settings Window for FaceAttend
Provides UI for model selection and performance monitoring
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logger import get_module_logger
from src.utils.exceptions import FaceRecognitionError

# Import recognition factory
try:
    from src.recognition import (
        get_recognition_factory, 
        RecognitionMode, 
        FACTORY_AVAILABLE,
        CLASSICAL_AVAILABLE,
        DL_AVAILABLE
    )
except ImportError:
    FACTORY_AVAILABLE = False


class RecognitionSettingsWindow:
    """Window for recognition engine settings and monitoring"""
    
    def __init__(self, parent=None):
        self.logger = get_module_logger("RecognitionSettings")
        self.parent = parent
        self.window = None
        self.is_window_open = False
        
        # Recognition factory
        self.factory = None
        if FACTORY_AVAILABLE:
            try:
                self.factory = get_recognition_factory()
            except Exception as e:
                self.logger.error(f"Failed to get recognition factory: {e}")
        
        # Monitoring variables
        self.monitoring_active = False
        self.monitor_thread = None
        
        # UI variables
        self.current_mode_var = None
        self.status_vars = {}
        self.performance_vars = {}
    
    def show_window(self):
        """Show the recognition settings window"""
        if self.is_window_open and self.window:
            self.window.lift()
            self.window.focus()
            return
        
        self.create_window()
        self.is_window_open = True
    
    def create_window(self):
        """Create the recognition settings window"""
        self.window = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.window.title("Recognition Engine Settings")
        self.window.geometry("700x600")
        self.window.resizable(True, True)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_engine_selection_tab(notebook)
        self.create_performance_monitoring_tab(notebook)
        self.create_system_status_tab(notebook)
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Start monitoring if factory is available
        if self.factory:
            self.start_monitoring()
        
        self.logger.info("Recognition settings window created")
    
    def create_engine_selection_tab(self, notebook):
        """Create engine selection tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="Engine Selection")
        
        # Title
        title_label = ttk.Label(
            tab_frame, 
            text="Recognition Engine Configuration", 
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        if not FACTORY_AVAILABLE:
            error_label = ttk.Label(
                tab_frame,
                text="Recognition factory not available.\nInstall deep learning dependencies for full functionality.",
                foreground="red",
                justify=tk.CENTER
            )
            error_label.pack(pady=20)
            return
        
        # Engine availability status
        availability_frame = ttk.LabelFrame(tab_frame, text="Engine Availability", padding=10)
        availability_frame.pack(fill=tk.X, padx=10, pady=10)
        
        classical_status = "✅ Available" if CLASSICAL_AVAILABLE else "❌ Not Available"
        dl_status = "✅ Available" if DL_AVAILABLE else "❌ Not Available"
        
        ttk.Label(availability_frame, text=f"Classical LBPH: {classical_status}").pack(anchor=tk.W)
        ttk.Label(availability_frame, text=f"Deep Learning ArcFace: {dl_status}").pack(anchor=tk.W)
        
        # Recognition mode selection
        mode_frame = ttk.LabelFrame(tab_frame, text="Recognition Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.current_mode_var = tk.StringVar()
        if self.factory:
            self.current_mode_var.set(self.factory.get_recognition_mode().value)
        
        # Mode radio buttons
        mode_options = []
        if CLASSICAL_AVAILABLE:
            mode_options.append(("Classical LBPH Only", RecognitionMode.CLASSICAL.value))
        if DL_AVAILABLE:
            mode_options.append(("Deep Learning ArcFace Only", RecognitionMode.DEEP_LEARNING.value))
        if CLASSICAL_AVAILABLE and DL_AVAILABLE:
            mode_options.append(("Hybrid (Best Result)", RecognitionMode.HYBRID.value))
        mode_options.append(("Auto Selection", RecognitionMode.AUTO.value))
        
        for text, value in mode_options:
            radio = ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.current_mode_var,
                value=value,
                command=self.on_mode_change
            )
            radio.pack(anchor=tk.W, pady=2)
        
        # Engine information
        info_frame = ttk.LabelFrame(tab_frame, text="Engine Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollable text widget for engine info
        info_text_frame = ttk.Frame(info_frame)
        info_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.engine_info_text = tk.Text(
            info_text_frame,
            height=8,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        info_scrollbar = ttk.Scrollbar(info_text_frame, orient=tk.VERTICAL, command=self.engine_info_text.yview)
        self.engine_info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.engine_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh button
        refresh_btn = ttk.Button(
            info_frame,
            text="🔄 Refresh Engine Information",
            command=self.refresh_engine_info
        )
        refresh_btn.pack(pady=5)
        
        # Initial info load
        self.refresh_engine_info()
    
    def create_performance_monitoring_tab(self, notebook):
        """Create performance monitoring tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="Performance Monitor")
        
        # Title
        title_label = ttk.Label(
            tab_frame, 
            text="Real-time Performance Monitoring", 
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        if not FACTORY_AVAILABLE:
            error_label = ttk.Label(
                tab_frame,
                text="Performance monitoring requires recognition factory.",
                foreground="red"
            )
            error_label.pack(pady=20)
            return
        
        # Performance metrics frame
        metrics_frame = ttk.LabelFrame(tab_frame, text="Engine Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create performance display for each engine
        self.performance_vars = {}
        
        if CLASSICAL_AVAILABLE:
            classical_frame = ttk.LabelFrame(metrics_frame, text="Classical LBPH Engine", padding=5)
            classical_frame.pack(fill=tk.X, pady=5)
            self.create_performance_display(classical_frame, "classical")
        
        if DL_AVAILABLE:
            dl_frame = ttk.LabelFrame(metrics_frame, text="Deep Learning ArcFace Engine", padding=5)
            dl_frame.pack(fill=tk.X, pady=5)
            self.create_performance_display(dl_frame, "deep_learning")
        
        # Overall statistics
        overall_frame = ttk.LabelFrame(tab_frame, text="Overall Statistics", padding=10)
        overall_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.overall_stats_text = tk.Text(
            overall_frame,
            height=6,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.overall_stats_text.pack(fill=tk.X)
        
        # Control buttons
        control_frame = ttk.Frame(tab_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            control_frame,
            text="📊 Refresh Metrics",
            command=self.refresh_performance_metrics
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="🧹 Clear Metrics",
            command=self.clear_performance_metrics
        ).pack(side=tk.LEFT, padx=5)
    
    def create_performance_display(self, parent, engine_name):
        """Create performance display for a specific engine"""
        self.performance_vars[engine_name] = {}
        
        # Metrics grid
        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill=tk.X)
        
        # Total recognitions
        ttk.Label(metrics_frame, text="Total Recognitions:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.performance_vars[engine_name]['total'] = tk.StringVar(value="0")
        ttk.Label(metrics_frame, textvariable=self.performance_vars[engine_name]['total']).grid(row=0, column=1, sticky=tk.W)
        
        # Success rate
        ttk.Label(metrics_frame, text="Success Rate:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.performance_vars[engine_name]['success_rate'] = tk.StringVar(value="0.0%")
        ttk.Label(metrics_frame, textvariable=self.performance_vars[engine_name]['success_rate']).grid(row=1, column=1, sticky=tk.W)
        
        # Average confidence
        ttk.Label(metrics_frame, text="Avg Confidence:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.performance_vars[engine_name]['confidence'] = tk.StringVar(value="0.000")
        ttk.Label(metrics_frame, textvariable=self.performance_vars[engine_name]['confidence']).grid(row=2, column=1, sticky=tk.W)
        
        # Average processing time
        ttk.Label(metrics_frame, text="Avg Processing Time:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.performance_vars[engine_name]['processing_time'] = tk.StringVar(value="0.000s")
        ttk.Label(metrics_frame, textvariable=self.performance_vars[engine_name]['processing_time']).grid(row=3, column=1, sticky=tk.W)
        
        # Last recognition time
        ttk.Label(metrics_frame, text="Last Recognition:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.performance_vars[engine_name]['last_time'] = tk.StringVar(value="0.000s")
        ttk.Label(metrics_frame, textvariable=self.performance_vars[engine_name]['last_time']).grid(row=4, column=1, sticky=tk.W)
    
    def create_system_status_tab(self, notebook):
        """Create system status tab"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text="System Status")
        
        # Title
        title_label = ttk.Label(
            tab_frame, 
            text="Recognition System Status", 
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # System status display
        status_frame = ttk.LabelFrame(tab_frame, text="Current Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.system_status_text = tk.Text(
            status_frame,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.system_status_text.yview)
        self.system_status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.system_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh button
        refresh_status_btn = ttk.Button(
            status_frame,
            text="🔄 Refresh System Status",
            command=self.refresh_system_status
        )
        refresh_status_btn.pack(pady=5)
        
        # Initial status load
        self.refresh_system_status()
    
    def create_status_bar(self, parent):
        """Create status bar at bottom of window"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Recognition settings ready")
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Last update time
        self.last_update_var = tk.StringVar()
        update_label = ttk.Label(status_frame, textvariable=self.last_update_var)
        update_label.pack(side=tk.RIGHT)
    
    def on_mode_change(self):
        """Handle recognition mode change"""
        if not self.factory:
            return
        
        try:
            new_mode_value = self.current_mode_var.get()
            new_mode = RecognitionMode(new_mode_value)
            
            self.factory.set_recognition_mode(new_mode)
            self.status_var.set(f"Recognition mode changed to: {new_mode.value}")
            self.logger.info(f"Recognition mode changed to: {new_mode.value}")
            
            # Refresh engine info
            self.refresh_engine_info()
            
        except Exception as e:
            error_msg = f"Failed to change recognition mode: {str(e)}"
            self.status_var.set(error_msg)
            self.logger.error(error_msg)
            messagebox.showerror("Mode Change Error", error_msg)
    
    def refresh_engine_info(self):
        """Refresh engine information display"""
        if not self.factory:
            self.engine_info_text.delete(1.0, tk.END)
            self.engine_info_text.insert(tk.END, "Recognition factory not available")
            return
        
        try:
            engine_info = self.factory.get_engine_info()
            available_engines = self.factory.get_available_engines()
            current_mode = self.factory.get_recognition_mode()
            
            info_text = f"Current Recognition Mode: {current_mode.value}\n"
            info_text += f"Available Engines: {', '.join(available_engines)}\n\n"
            
            for engine_name, info in engine_info.items():
                info_text += f"=== {engine_name.upper()} ENGINE ===\n"
                for key, value in info.items():
                    info_text += f"  {key}: {value}\n"
                info_text += "\n"
            
            self.engine_info_text.delete(1.0, tk.END)
            self.engine_info_text.insert(tk.END, info_text)
            
        except Exception as e:
            error_msg = f"Error refreshing engine info: {str(e)}"
            self.engine_info_text.delete(1.0, tk.END)
            self.engine_info_text.insert(tk.END, error_msg)
            self.logger.error(error_msg)
    
    def refresh_performance_metrics(self):
        """Refresh performance metrics display"""
        if not self.factory:
            return
        
        try:
            metrics = self.factory.get_performance_metrics()
            
            for engine_name, perf in metrics.items():
                if engine_name in self.performance_vars:
                    vars_dict = self.performance_vars[engine_name]
                    vars_dict['total'].set(str(perf.total_recognitions))
                    vars_dict['success_rate'].set(f"{perf.success_rate:.1f}%")
                    vars_dict['confidence'].set(f"{perf.average_confidence:.3f}")
                    vars_dict['processing_time'].set(f"{perf.average_processing_time:.3f}s")
                    vars_dict['last_time'].set(f"{perf.last_recognition_time:.3f}s")
            
            # Update overall statistics
            total_recognitions = sum(perf.total_recognitions for perf in metrics.values())
            total_successful = sum(perf.successful_recognitions for perf in metrics.values())
            
            overall_success_rate = (total_successful / total_recognitions * 100) if total_recognitions > 0 else 0
            
            overall_text = f"Overall System Performance:\n"
            overall_text += f"  Total Recognitions: {total_recognitions}\n"
            overall_text += f"  Total Successful: {total_successful}\n"
            overall_text += f"  Overall Success Rate: {overall_success_rate:.1f}%\n"
            overall_text += f"  Active Engines: {len(metrics)}\n"
            overall_text += f"  Last Update: {datetime.now().strftime('%H:%M:%S')}"
            
            self.overall_stats_text.delete(1.0, tk.END)
            self.overall_stats_text.insert(tk.END, overall_text)
            
            self.last_update_var.set(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            error_msg = f"Error refreshing performance metrics: {str(e)}"
            self.logger.error(error_msg)
    
    def refresh_system_status(self):
        """Refresh system status display"""
        if not self.factory:
            self.system_status_text.delete(1.0, tk.END)
            self.system_status_text.insert(tk.END, "Recognition factory not available")
            return
        
        try:
            status = self.factory.get_system_status()
            
            status_text = f"=== FACEATTEND RECOGNITION SYSTEM STATUS ===\n\n"
            status_text += f"Current Mode: {status['current_mode']}\n"
            status_text += f"Available Engines: {', '.join(status['available_engines'])}\n"
            status_text += f"Classical Available: {status['classical_available']}\n"
            status_text += f"Deep Learning Available: {status['deep_learning_available']}\n\n"
            
            status_text += "=== ENGINE INFORMATION ===\n"
            for engine, info in status['engine_info'].items():
                status_text += f"\n{engine.upper()} Engine:\n"
                for key, value in info.items():
                    status_text += f"  {key}: {value}\n"
            
            status_text += "\n=== PERFORMANCE SUMMARY ===\n"
            for engine, perf in status['performance_metrics'].items():
                status_text += f"\n{engine.upper()}:\n"
                status_text += f"  Recognitions: {perf['total_recognitions']}\n"
                status_text += f"  Success Rate: {perf['success_rate']:.1f}%\n"
                status_text += f"  Avg Confidence: {perf['avg_confidence']:.3f}\n"
                status_text += f"  Avg Time: {perf['avg_processing_time']:.3f}s\n"
            
            status_text += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.system_status_text.delete(1.0, tk.END)
            self.system_status_text.insert(tk.END, status_text)
            
        except Exception as e:
            error_msg = f"Error refreshing system status: {str(e)}"
            self.system_status_text.delete(1.0, tk.END)
            self.system_status_text.insert(tk.END, error_msg)
            self.logger.error(error_msg)
    
    def clear_performance_metrics(self):
        """Clear performance metrics (if supported by factory)"""
        if not self.factory:
            return
        
        try:
            # Note: This would require adding a clear_metrics method to the factory
            # For now, just refresh to show current state
            self.refresh_performance_metrics()
            self.status_var.set("Performance metrics refreshed")
            
        except Exception as e:
            error_msg = f"Error clearing metrics: {str(e)}"
            self.status_var.set(error_msg)
            self.logger.error(error_msg)
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                if self.window and self.window.winfo_exists():
                    # Schedule UI updates in main thread
                    self.window.after_idle(self.refresh_performance_metrics)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                break
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_monitoring()
        self.is_window_open = False
        if self.window:
            self.window.destroy()
        self.logger.info("Recognition settings window closed")


def open_recognition_settings_window(parent=None):
    """Open the recognition settings window"""
    window = RecognitionSettingsWindow(parent)
    window.show_window()
    return window


# Test function
def test_recognition_settings_window():
    """Test the recognition settings window"""
    root = tk.Tk()
    root.withdraw()  # Hide root window
    
    window = open_recognition_settings_window()
    root.mainloop()


if __name__ == "__main__":
    test_recognition_settings_window()