#!/usr/bin/env python3
"""
Restore original InsightFace functionality
Run this when dependencies are properly installed
"""

from pathlib import Path

def restore_original():
    try:
        recognizer_file = Path('src/recognition/realtime_recognizer.py')
        
        with open(recognizer_file, 'r') as f:
            content = f.read()
        
        # Restore original import
        content = content.replace(
            """# Original import replaced with mock for testing
# from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer
from src.recognition.mock_recognizer import SimpleInsightFaceRecognizer""",
            'from src.recognition.simple_insightface_recognizer import SimpleInsightFaceRecognizer'
        )
        
        with open(recognizer_file, 'w') as f:
            f.write(content)
        
        print("✅ Restored original InsightFace functionality")
        return True
        
    except Exception as e:
        print(f"❌ Error restoring: {e}")
        return False

if __name__ == "__main__":
    restore_original()
