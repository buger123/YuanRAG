"""Ingestion package — format detect, parsers, chunkers, pipeline."""
from .format_detect import Format, detect

__all__ = ["Format", "detect"]
