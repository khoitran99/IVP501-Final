"""
Model downloader and management system for deep learning models.
Handles automatic downloading, validation, and caching of pre-trained models.
"""

import os
import hashlib
import requests
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

class ModelDownloader:
    """Manages downloading and validation of deep learning models."""
    
    def __init__(self, config_path: str = None):
        """Initialize model downloader with configuration."""
        self.base_dir = Path(__file__).parent
        self.config_path = config_path or self.base_dir.parent / "config" / "models.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Create default config if not exists
            default_config = self._create_default_config()
            self._save_config(default_config)
            return default_config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default model configuration."""
        return {
            'models': {
                'detection': {
                    'mtcnn': {
                        'url': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt',
                        'file': 'mtcnn_weights.pt',
                        'sha256': 'placeholder_hash',
                        'description': 'MTCNN face detection model'
                    }
                },
                'recognition': {
                    'arcface_r100': {
                        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx',
                        'file': 'arcface_r100_v1.onnx',
                        'sha256': 'placeholder_hash',
                        'feature_dim': 512,
                        'description': 'ArcFace ResNet100 recognition model'
                    },
                    'arcface_mobilenet': {
                        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_mobilefacenet.onnx',
                        'file': 'arcface_mobilefacenet.onnx',
                        'sha256': 'placeholder_hash',
                        'feature_dim': 128,
                        'description': 'ArcFace MobileFaceNet recognition model'
                    }
                }
            }
        }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file."""
        os.makedirs(self.config_path.parent, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def download_model(self, model_type: str, model_name: str, force_download: bool = False) -> str:
        """
        Download a specific model.
        
        Args:
            model_type: Type of model ('detection' or 'recognition')
            model_name: Name of the model
            force_download: Whether to force re-download if file exists
            
        Returns:
            Path to downloaded model file
        """
        if model_type not in self.config['models']:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_name not in self.config['models'][model_type]:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.config['models'][model_type][model_name]
        model_dir = self.base_dir / model_type / model_name
        model_path = model_dir / model_config['file']
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if model already exists and is valid
        if model_path.exists() and not force_download:
            if self._validate_model(model_path, model_config.get('sha256')):
                print(f"Model {model_name} already exists and is valid")
                return str(model_path)
            else:
                print(f"Model {model_name} exists but validation failed, re-downloading...")
        
        # Download model
        print(f"Downloading {model_name} from {model_config['url']}...")
        self._download_file(model_config['url'], model_path)
        
        # Validate downloaded model
        if not self._validate_model(model_path, model_config.get('sha256')):
            print(f"Warning: Model validation failed for {model_name}")
        
        return str(model_path)
    
    def _download_file(self, url: str, destination: Path) -> None:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    def _validate_model(self, model_path: Path, expected_hash: Optional[str] = None) -> bool:
        """Validate model file integrity."""
        if not model_path.exists():
            return False
        
        # Basic file size check
        if model_path.stat().st_size == 0:
            return False
        
        # SHA256 validation if hash is provided
        if expected_hash and expected_hash != 'placeholder_hash':
            actual_hash = self._calculate_sha256(model_path)
            return actual_hash == expected_hash
        
        return True
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def download_all_models(self) -> Dict[str, str]:
        """Download all configured models."""
        downloaded_models = {}
        
        for model_type in self.config['models']:
            for model_name in self.config['models'][model_type]:
                try:
                    path = self.download_model(model_type, model_name)
                    downloaded_models[f"{model_type}/{model_name}"] = path
                except Exception as e:
                    print(f"Failed to download {model_type}/{model_name}: {e}")
        
        return downloaded_models
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models in configuration."""
        return self.config['models']
    
    def get_model_path(self, model_type: str, model_name: str) -> Optional[str]:
        """Get path to model if it exists locally."""
        if model_type not in self.config['models']:
            return None
        
        if model_name not in self.config['models'][model_type]:
            return None
        
        model_config = self.config['models'][model_type][model_name]
        model_dir = self.base_dir / model_type / model_name
        model_path = model_dir / model_config['file']
        
        return str(model_path) if model_path.exists() else None


def main():
    """CLI interface for model downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and manage deep learning models')
    parser.add_argument('--download-all', action='store_true', help='Download all configured models')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--type', help='Model type (detection/recognition)')
    parser.add_argument('--name', help='Model name')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.list:
        models = downloader.list_available_models()
        print("Available models:")
        for model_type, model_dict in models.items():
            print(f"\n{model_type.upper()}:")
            for name, config in model_dict.items():
                print(f"  - {name}: {config.get('description', 'No description')}")
    
    elif args.download_all:
        print("Downloading all models...")
        downloaded = downloader.download_all_models()
        print(f"\nDownloaded {len(downloaded)} models:")
        for model, path in downloaded.items():
            print(f"  - {model}: {path}")
    
    elif args.type and args.name:
        try:
            path = downloader.download_model(args.type, args.name, args.force)
            print(f"Model downloaded to: {path}")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()