"""A-share data processing utilities based on AkShare."""

from ashare_data_processing.providers import AkShareProvider, SyntheticProvider, build_provider
from ashare_data_processing.quality import quality_report
from ashare_data_processing.storage import CsvDataStore

__all__ = [
    "AkShareProvider",
    "SyntheticProvider",
    "CsvDataStore",
    "build_provider",
    "quality_report",
]
