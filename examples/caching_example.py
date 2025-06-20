"""
Example usage of Predictpy with optimized caching.
"""
import time
from predictpy import Predictpy, calculate_optimal_cache_size

def main():
    # Initialize with optimal cache sizes
    cache_sizes = calculate_optimal_cache_size()
    print(f"Using cache sizes: {cache_sizes}")
    
    predictor = Predictpy(
        config={'cache_config': cache_sizes}
    )
    
    # Monitor cache performance
    print("Running prediction tests...")
    for i in range(100):
        # Mix of cached and new queries
        if i % 3 == 0:
            predictor.predict("I want to")  # Should be cached
        else:
            predictor.predict(f"Test phrase {i}")  # New query
    
    # Check cache performance
    cache_info = predictor.cache_info
    print(f"Cache hit rate: {cache_info['predict_cache']['hit_rate']:.2%}")
    print(f"Cache size: {cache_info['predict_cache']['currsize']}/{cache_info['predict_cache']['maxsize']}")
    
    # Test invalidation
    print("\nTesting cache invalidation...")
    for i in range(60):
        predictor.select("I like", "to")
    
    # Check modification counters
    print(f"Modifications since clear: {predictor.cache_info['modifications_since_clear']}")
    
    # Force clear all caches
    print("\nForcing cache clear...")
    predictor.clear_all_caches()
    print(f"Cache after clear - size: {predictor.cache_info['predict_cache']['currsize']}")

if __name__ == "__main__":
    main()
