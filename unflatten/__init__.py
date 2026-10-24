"""slide-unflatten: image-only slides -> editable PPTX."""
from .config import Config, PRESETS, LANGUAGE_PROFILES, MODELS
from .pipeline import convert, estimate, rebuild

__all__ = ["Config", "PRESETS", "LANGUAGE_PROFILES", "MODELS", "convert", "estimate", "rebuild", "__version__"]

__version__ = "0.2.0"
