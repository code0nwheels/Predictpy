"""
Utility functions for Predictpy
"""
import logging
from typing import Dict

def calculate_optimal_cache_size(available_memory_mb: int = None) -> Dict[str, int]:
    """
    Calculate optimal cache sizes based on available memory.
    
    Args:
        available_memory_mb: Optional amount of memory to use (in MB). 
                            If None, automatically detects available memory.
    
    Returns:
        Dictionary with optimal cache sizes for different caches
        
    The function allocates cache entries based on estimated memory usage:
    - Prediction cache: ~200 bytes per entry
    - Completion cache: ~2KB per entry
    """
    try:
        import psutil
        
        if available_memory_mb is None:
            # Get system memory
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Use 5% of available memory for caching
        cache_memory_mb = min(available_memory_mb * 0.05, 200)  # Cap at 200MB
        
        # Estimate memory per cache entry
        # Prediction cache: ~200 bytes per entry
        # Completion cache: ~2KB per entry
        
        return {
            'predict_size': min(int(cache_memory_mb * 1024 * 1024 * 0.8 / 200), 8192),
            'completion_size': min(int(cache_memory_mb * 1024 * 1024 * 0.2 / 2048), 512)
        }
    except ImportError:
        logging.warning("psutil not available, using default cache sizes")
        # Default conservative settings
        return {
            'predict_size': 1000, 
            'completion_size': 100
        }
    except Exception as e:
        logging.warning(f"Error calculating cache sizes: {e}, using defaults")
        return {
            'predict_size': 1000, 
            'completion_size': 100
        }
