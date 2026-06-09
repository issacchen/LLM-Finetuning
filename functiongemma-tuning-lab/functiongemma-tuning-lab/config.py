import os
from pathlib import Path
from typing import Final, Optional, List
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    """
    Central configuration class.
    """
    # Directory Setup
    ARTIFACTS_DIR: Final[Path] = Path("artifacts")
    OUTPUT_DIR: Final[Path] = ARTIFACTS_DIR.joinpath("functiongemma-tuning-lab-demo")
    
    # Model & Data
    HF_TOKEN: Final[Optional[str]] = os.getenv('HF_TOKEN')
    
    # Model Configuration
    # Mutable: User can change this in the UI
    MODEL_NAME: str = 'google/functiongemma-270m-it'

    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: [
        'google/functiongemma-270m-it'
    ])

    DEFAULT_DATASET: Final[str] = 'bebechien/SimpleToolCalling'
    
    def __post_init__(self):
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
