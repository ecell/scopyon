import importlib.metadata

from .base import *
from .config import *
from .image import *
from .sampling import *
from .sampling2 import *
from . import constants
from . import analysis

__all__ = [
    "EnvironSettings", "EPIFMSimulator",
    "form_image", "generate_images", "create_simulator",
    "Configuration", "DefaultConfiguration",
    "Image", "Video",
    "sample_inputs",
    "sample",
    "constants", "analysis"
    ]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
