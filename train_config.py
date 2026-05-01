"""
Training configuration with environment profiles for home GPU and NRP A100.
Supports automatic GPU detection and profile selection.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class TrainingConfig:
    """Base training configuration"""
    data_file: str = "input/train.jsonl"
    output_dir: str = "checkpoints"
    max_samples: Optional[int] = None  # None for full dataset
    num_epochs: int = 1  # Short training for testing
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    seed: int = 42
    # LoRA / CUDA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 50
    
    def validate(self):
        """Validate configuration"""
        if not Path(self.data_file).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'data_file': self.data_file,
            'output_dir': self.output_dir,
            'max_samples': self.max_samples,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'use_4bit': self.use_4bit,
            'bf16': self.bf16,
            'gradient_checkpointing': self.gradient_checkpointing,
            'max_length': self.max_length,
            'warmup_steps': self.warmup_steps,
            'save_steps': self.save_steps,
            'logging_steps': self.logging_steps,
        }


# Environment profiles optimized for different hardware
TRAINING_PROFILES: Dict[str, Dict[str, Any]] = {
    "home_dev": {
        "description": "Home GPU (12GB VRAM) - Fast iteration and testing",
        "batch_size": 1,
        "gradient_accumulation_steps": 16,  # Effective batch = 16
        "gradient_checkpointing": True,      # Save ~2GB VRAM
        "max_length": 512,                   # Conservative sequence length
        "lora_r": 8,                         # Smaller adapters
        "lora_alpha": 16,
        "use_4bit": True,
        "num_epochs": 1,
        "max_samples": 100,                  # Quick validation
        "warmup_steps": 50,
        "save_steps": 50,
        "logging_steps": 10,
        "learning_rate": 2e-4,
    },
    "home_dev_batch2_test": {
        "description": "Home GPU (12GB VRAM) - Batch size 2 pipeline testing (ultra-conservative for OOM avoidance)",
        "batch_size": 2,                     # Test batching code path
        "gradient_accumulation_steps": 1,   # Minimal: effective batch = 2 (ultra-conservative)
        "gradient_checkpointing": True,      # Save ~2GB VRAM
        "max_length": 64,                   # Aggressive reduction (from 256) to prevent loss OOM
        "lora_r": 4,                         # Minimal LoRA rank to save adapter memory
        "lora_alpha": 8,
        "use_4bit": True,
        "num_epochs": 1,
        "max_samples": 10,                   # Minimal for quick validation (was 30)
        "warmup_steps": 2,
        "save_steps": 5,
        "logging_steps": 1,
        "learning_rate": 2e-4,
    },
    "nrp_a100": {
        "description": "NRP A100 24GB - Production training (memory optimized)",
        "batch_size": 1,                     # Use 1 for variable image sizes
        "gradient_accumulation_steps": 16,   # Effective batch = 16 (same as home)
        "gradient_checkpointing": True,      # Required for 24GB
        "max_length": 1024,                   # Reduced for 24GB VRAM
        "lora_r": 16,                         # Lower rank for memory (same as home_dev)
        "lora_alpha": 32,
        "use_4bit": True,                    # Keep compatible with home inference
        "num_epochs": 3,
        "max_samples": None,                 # Full dataset
        "warmup_steps": 100,
        "save_steps": 200,
        "logging_steps": 20,
        "learning_rate": 2e-4,
    },
    "nrp_a100_32gb": {
        "description": "NRP A100 32GB - Production training (high quality)",
        "batch_size": 16,                     
        "gradient_accumulation_steps": 3,   # Effective batch = 48
        "gradient_checkpointing": False,     
        "max_length": 2048,                  # Full context window
        "lora_r": 64,                        # Higher rank for better adapters
        "lora_alpha": 128,
        "use_4bit": True,                    # Keep compatible with home inference
        "num_epochs": 3,
        "max_samples": None,                 # Full dataset
        "warmup_steps": 100,
        "save_steps": 200,
        "logging_steps": 20,
        "learning_rate": 2e-4,
    },
    "mac_mps_128gb": {
        "description": "Apple Silicon M5 with 128GB RAM - High-capacity MPS training",
        "batch_size": 1,                     # Conservative for variable image sizes
        "gradient_accumulation_steps": 16,   # Effective batch = 16
        "gradient_checkpointing": False,     # Not needed with 128GB system RAM
        "max_length": 1024,                  # Full context window
        "lora_r": 16,                        # Higher rank for better adapters
        "lora_alpha": 32,
        "use_4bit": False,                   # MPS may not support 4-bit quantization well, use full precision
        "bf16": False,                     # MPS supports bfloat16, but may have issues with some models - use float16 for compatibility              
        "num_epochs": 3,
        "max_samples": None,                 # Full dataset
        "warmup_steps": 100,
        "save_steps": 200,
        "logging_steps": 20,
        "learning_rate": 2e-4,
    }

}

def get_gpu_memory_gb() -> Optional[float]:
    """Detect GPU VRAM in GB, return None if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)  # Convert to GB
    except Exception:
        pass
    return None


def detect_device() -> tuple:
    """
    Detect available hardware device and preferred data type.
    Priority: MPS (Apple Silicon) > CUDA > CPU (with warning).
    
    Returns:
        tuple of (device_name, dtype_name)
    """
    try:
        import torch
        
        # Check MPS first (Apple Silicon)
        if torch.backends.mps.is_available():
            dtype = "bfloat16" if hasattr(torch.backends.mps, 'is_bf16_supported') and torch.backends.mps.is_bf16_supported() else "float16"
            print(f"🔍 Detected MPS (Apple Silicon) with {dtype} support")
            return ("mps", dtype)
        
        # Check CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            print(f"🔍 Detected CUDA GPU ({props.name}) with {vram_gb:.1f}GB VRAM, using {dtype}")
            return ("cuda", dtype)
        
    except Exception:
        pass
    
    # Fallback to CPU (with warning)
    print("⚠️  No GPU detected - falling back to CPU. Training will be slow!")
    return ("cpu", "float32")


def detect_profile() -> str:
    """Auto-detect appropriate profile based on available hardware."""
    device, dtype = detect_device()
    
    # Map device to appropriate profile
    if device == "mps":
        print("   → Suggesting 'mac_mps_128gb' profile (Apple Silicon MPS optimized)")
        return "mac_mps_128gb"
    elif device == "cuda":
        vram_gb = get_gpu_memory_gb()
        
        if vram_gb is None:
            print("   → Suggesting 'home_dev' profile (12GB VRAM optimized)")
            return "home_dev"
        
        print(f"🔍 Detected GPU with {vram_gb:.1f}GB VRAM")
        
        # Select best profile based on VRAM: <20GB = home, 20-28GB = nrp_a100, >=28GB = nrp_a100_32gb
        if vram_gb < 20:
            print("   → Suggesting 'home_dev' profile (12GB VRAM optimized)")
            return "home_dev"
        elif vram_gb < 28:
            print("   → Suggesting 'nrp_a100' profile (24GB VRAM optimized)")
            return "nrp_a100"
        else:
            print("   → Suggesting 'nrp_a100_32gb' profile (32GB VRAM high quality)")
            return "nrp_a100_32gb"
    else:
        print("   → Suggesting 'home_dev' profile (CPU fallback)")
        return "home_dev"


def load_profile(profile_name: str) -> Dict[str, Any]:
    """Load a training profile by name."""
    if profile_name not in TRAINING_PROFILES:
        available = ", ".join(TRAINING_PROFILES.keys())
        raise ValueError(
            f"Unknown profile: {profile_name}. Available profiles: {available}"
        )
    
    profile = TRAINING_PROFILES[profile_name]
    print(f"\n📋 Loading profile: {profile_name}")
    print(f"   {profile['description']}")
    return profile.copy()


def apply_profile_to_config(config: TrainingConfig, profile: Dict[str, Any]) -> TrainingConfig:
    """Apply profile settings to a TrainingConfig instance."""
    for key, value in profile.items():
        if key == "description":
            continue  # Skip description field
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def load_training_data(data_file: str, max_samples: Optional[int] = None):
    """
    Load training data from JSONL file.
    
    Args:
        data_file: Path to JSONL training file
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of training examples
    """
    examples = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {i}: {e}")
                continue
    
    return examples


def print_training_summary(config: TrainingConfig, num_examples: int):
    """Print training configuration summary"""
    print("=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Data file: {config.data_file}")
    print(f"Number of examples: {num_examples}")
    print(f"Output directory: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA rank: {config.lora_r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"Max sequence length: {config.max_length}")
    print("=" * 60)

