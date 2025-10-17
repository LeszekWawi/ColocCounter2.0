"""
Processing module - Batch processing and result management
"""

from .processing_result import ProcessingResult
from .result_collector import ThreadSafeResultCollector
from .batch_processor import BatchProcessor

__all__ = ['ProcessingResult', 'ThreadSafeResultCollector', 'BatchProcessor']
