# Configuration

Comprehensive guide to configuring Predictpy for different use cases and environments.

## Configuration Methods

### Constructor Configuration

The most common way to configure Predictpy:

```python
from predictpy import Predictpy

predictor = Predictpy(
    config=None,                    # Config file path or dict
    db_path=None,                   # Database location
    auto_train=True,                # Auto-train on first use
    training_size="medium",         # Training data size
    use_semantic=True              # Enable semantic features
)
```

### Configuration Files

Create reusable configuration files in JSON format:

```json
{
    "db_path": "/custom/path/predictpy.db",
    "auto_train": true,
    "target_sentences": 25000,
    "use_semantic": true,
    "semantic_model": "all-MiniLM-L6-v2",
    "max_predictions": 10,
    "learning": {
        "enable_personal": true,
        "weight_personal": 0.3,
        "cleanup_days": 90
    },
    "cache_config": {
        "predict_size": 2000,
        "completion_size": 200,
        "ttl_seconds": 3600
    }
}
```

Load from file:

```python
predictor = Predictpy(config="/path/to/config.json")

# Or load manually
import json
with open('config.json', 'r') as f:
    config = json.load(f)
predictor = Predictpy(config=config)
```

---

## Core Configuration Options

### Database Configuration

```python
# Default location (~/.predictpy/)
predictor = Predictpy()

# Custom database path
predictor = Predictpy(db_path="/custom/path/predictions.db")

# In-memory database (testing)
predictor = Predictpy(db_path=":memory:")

# Shared database for multiple users
predictor = Predictpy(db_path="/shared/team_predictions.db")
```

### Training Configuration

```python
# Training sizes
predictor = Predictpy(training_size="small")    # 1,000 sentences
predictor = Predictpy(training_size="medium")   # 10,000 sentences  
predictor = Predictpy(training_size="large")    # 50,000 sentences

# Disable auto-training
predictor = Predictpy(auto_train=False)

# Custom training size
config = {"target_sentences": 25000}
predictor = Predictpy(config=config)
```

### Semantic Configuration

```python
# Enable/disable semantic features
predictor = Predictpy(use_semantic=True)   # Default
predictor = Predictpy(use_semantic=False)  # Word-only prediction

# Custom semantic model
config = {
    "semantic_model": "all-mpnet-base-v2",  # More accurate, larger
    "semantic_db_path": "/custom/chroma/path"
}
predictor = Predictpy(config=config, use_semantic=True)
```

---

## Environment-Specific Configurations

### Development Environment

```python
# Fast startup for development
dev_config = {
    "db_path": "./dev_predictions.db",
    "auto_train": True,
    "target_sentences": 1000,      # Small training set
    "use_semantic": False,         # Skip semantic for speed
    "debug": True
}

predictor = Predictpy(config=dev_config)
```

### Production Environment

```python
# Optimized for production
prod_config = {
    "db_path": "/var/lib/predictpy/production.db", 
    "auto_train": False,           # Pre-trained database
    "use_semantic": True,
    "target_sentences": 50000,     # Large training set
    "semantic_model": "all-MiniLM-L6-v2",
    "max_predictions": 5,
    "learning": {
        "enable_personal": True,
        "cleanup_days": 30         # Aggressive cleanup
    }
}

predictor = Predictpy(config=prod_config)
```

### Testing Environment

```python
# Isolated testing setup
test_config = {
    "db_path": ":memory:",         # In-memory database
    "auto_train": False,           # No training
    "use_semantic": False,         # Faster tests
    "mock_predictions": True
}

predictor = Predictpy(config=test_config)
```

---

## Performance Tuning

### Memory Optimization

```python
# Low-memory configuration
low_memory_config = {
    "use_semantic": False,                    # Save ~100MB
    "target_sentences": 5000,                 # Smaller training
    "learning": {
        "max_personal_selections": 1000,      # Limit personal data
        "cleanup_days": 14                    # Frequent cleanup
    }
}

predictor = Predictpy(config=low_memory_config)
```

### Speed Optimization

```python
# Fast prediction configuration
fast_config = {
    "target_sentences": 1000,        # Quick training
    "use_semantic": False,           # Skip semantic processing
    "max_predictions": 3,            # Fewer predictions
    "cache_predictions": True,       # Enable caching
    "learning": {
        "batch_personal": True       # Batch personal updates
    }
}

predictor = Predictpy(config=fast_config)
```

### Accuracy Optimization

```python
# High-accuracy configuration
accuracy_config = {
    "target_sentences": 100000,              # Large training set
    "use_semantic": True,
    "semantic_model": "all-mpnet-base-v2",   # More accurate model
    "max_predictions": 10,                   # More options
    "learning": {
        "enable_personal": True,
        "weight_personal": 0.5,              # Strong personal weighting
        "min_selection_count": 3             # Require multiple selections
    }
}

predictor = Predictpy(config=accuracy_config)
```

---

## Advanced Configuration

### Custom Learning Parameters

```python
learning_config = {
    "learning": {
        "enable_personal": True,             # Enable personal learning
        "weight_personal": 0.3,              # Personal vs statistical weight
        "min_selection_count": 2,            # Min selections before learning
        "decay_factor": 0.95,                # Selection weight decay
        "context_window": 3,                 # Context word window
        "cleanup_days": 90,                  # Auto-cleanup threshold
        "max_personal_selections": 10000,    # Max personal data
        "batch_size": 100                    # Batch processing size
    }
}

predictor = Predictpy(config=learning_config)
```

### Semantic Model Configuration

```python
semantic_config = {
    "semantic": {
        "model_name": "all-MiniLM-L6-v2",    # Embedding model
        "db_path": "custom/chroma/path",      # ChromaDB location
        "collection_name": "user_thoughts",   # Collection name
        "embedding_dimension": 384,           # Model dimensions
        "max_patterns": 5000,                # Max stored patterns
        "similarity_threshold": 0.7,         # Similarity cutoff
        "batch_size": 32,                    # Embedding batch size
        "cache_embeddings": True             # Cache embeddings
    }
}

predictor = Predictpy(config=semantic_config, use_semantic=True)
```

### Multi-User Configuration

```python
# Configuration for multiple users
multiuser_config = {
    "db_path": "/shared/predictions/{user_id}.db",
    "user_isolation": True,
    "shared_base_model": True,
    "personal_learning": {
        "per_user_limit": 5000,
        "cross_user_learning": False,
        "privacy_mode": True
    },
    "semantic": {
        "shared_embeddings": True,
        "user_specific_patterns": True
    }
}

# Initialize for specific user
user_predictor = Predictpy(
    config=multiuser_config,
    db_path=f"/shared/predictions/{user_id}.db"
)
```

---

## Configuration Validation

### Validate Configuration

```python
def validate_config(config):
    """Validate configuration before use."""
    required_fields = ["db_path", "auto_train"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    if config.get("target_sentences", 0) < 100:
        raise ValueError("target_sentences must be at least 100")
    
    return True

# Use validation
try:
    validate_config(my_config)
    predictor = Predictpy(config=my_config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Configuration Templates

```python
# Template for different use cases
TEMPLATES = {
    "lightweight": {
        "target_sentences": 1000,
        "use_semantic": False,
        "max_predictions": 3
    },
    
    "balanced": {
        "target_sentences": 10000,
        "use_semantic": True,
        "max_predictions": 5
    },
    
    "comprehensive": {
        "target_sentences": 50000,
        "use_semantic": True,
        "semantic_model": "all-mpnet-base-v2",
        "max_predictions": 10
    }
}

# Use template
predictor = Predictpy(config=TEMPLATES["balanced"])
```

---

## Environment Variables

Set configuration through environment variables:

```bash
# Unix/Linux/Mac
export PREDICTPY_DB_PATH="/custom/path/predictions.db"
export PREDICTPY_TRAINING_SIZE="large"
export PREDICTPY_USE_SEMANTIC="true"

# Windows
set PREDICTPY_DB_PATH=C:\custom\path\predictions.db
set PREDICTPY_TRAINING_SIZE=large
set PREDICTPY_USE_SEMANTIC=true
```

Load from environment:

```python
import os

env_config = {
    "db_path": os.getenv("PREDICTPY_DB_PATH"),
    "training_size": os.getenv("PREDICTPY_TRAINING_SIZE", "medium"),
    "use_semantic": os.getenv("PREDICTPY_USE_SEMANTIC", "true").lower() == "true"
}

# Remove None values
env_config = {k: v for k, v in env_config.items() if v is not None}

predictor = Predictpy(config=env_config)
```

---

## Configuration Management

### Export Current Configuration

```python
# Export configuration for backup/sharing
predictor.export_config("backup_config.json")

# Manual export
import json

current_config = {
    "db_path": predictor.engine.predictor.db_path,
    "has_semantic": predictor.has_semantic,
    "stats": predictor.stats
}

with open("current_config.json", "w") as f:
    json.dump(current_config, f, indent=2)
```

### Configuration Migration

```python
def migrate_config(old_config, target_version="2.0"):
    """Migrate configuration to newer version."""
    
    # Version 1.x to 2.x migration
    if "personal_model" in old_config:
        old_config["learning"] = {
            "enable_personal": old_config.pop("personal_model", True)
        }
    
    # Add new defaults
    old_config.setdefault("use_semantic", True)
    old_config.setdefault("training_size", "medium")
    
    return old_config

# Apply migration
migrated_config = migrate_config(legacy_config)
predictor = Predictpy(config=migrated_config)
```

### Dynamic Configuration Updates

```python
class ConfigurablePredictor:
    def __init__(self, initial_config):
        self.config = initial_config
        self.predictor = Predictpy(config=initial_config)
    
    def update_config(self, new_config):
        """Update configuration dynamically."""
        self.config.update(new_config)
        
        # Some changes require reinitialization
        if "db_path" in new_config or "use_semantic" in new_config:
            self.predictor = Predictpy(config=self.config)
        
        return self.config
    
    def get_config(self):
        return self.config.copy()

# Usage
configurable = ConfigurablePredictor({"training_size": "small"})
configurable.update_config({"training_size": "large", "use_semantic": True})
```

---

## Best Practices

### Configuration Security

```python
# Don't store sensitive data in config
secure_config = {
    "db_path": "/secure/path/predictions.db",
    "use_encryption": True,  # If implemented
    "privacy_mode": True,
    "learning": {
        "anonymize_personal": True,
        "retention_days": 30
    }
}

# Use environment variables for sensitive paths
import os
secure_config["db_path"] = os.getenv("SECURE_DB_PATH", "/tmp/predictions.db")
```

### Configuration Documentation

```python
# Well-documented configuration
documented_config = {
    # Core settings
    "db_path": "/data/predictions.db",        # Main database location
    "auto_train": True,                       # Train on first use
    "training_size": "medium",                # Small/medium/large
    
    # Performance settings
    "use_semantic": True,                     # Enable AI completions
    "max_predictions": 5,                     # Limit prediction count
    
    # Learning settings
    "learning": {
        "enable_personal": True,              # Learn from selections
        "weight_personal": 0.3,               # Personal vs base weight
        "cleanup_days": 90                    # Auto-cleanup threshold
    },
    
    # Advanced settings
    "debug": False,                           # Debug mode
    "logging_level": "INFO"                   # Logging verbosity
}
```

### Configuration Testing

```python
def test_configuration(config):
    """Test configuration with actual usage."""
    try:
        # Test basic initialization
        predictor = Predictpy(config=config)
        
        # Test basic prediction
        suggestions = predictor.predict("hello")
        assert len(suggestions) > 0, "No predictions generated"
        
        # Test semantic features if enabled
        if config.get("use_semantic", False) and predictor.has_semantic:
            completions = predictor.predict_completion("hello")
            # Semantic might return empty for simple input
        
        # Test learning
        predictor.select("hello", "world")
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

# Test before deployment
if test_configuration(production_config):
    predictor = Predictpy(config=production_config)
else:
    predictor = Predictpy()  # Fallback to defaults
```

---

## Caching Configuration

Predictpy implements smart caching to improve performance of frequently accessed predictions and completions. You can configure cache behavior to optimize for your specific use case.

### Cache Settings

The following cache settings can be configured:

| Parameter | Description | Default | 
|-----------|-------------|---------|
| `predict_size` | Maximum number of prediction entries to cache | 1000 |
| `completion_size` | Maximum number of completion entries to cache | 100 |
| `ttl_seconds` | Cache time to live in seconds (cache invalidation period) | 3600 (1 hour) |

### Memory-Aware Cache Sizing

Predictpy provides a utility function to calculate optimal cache sizes based on available system memory:

```python
from predictpy import calculate_optimal_cache_size, Predictpy

# Get optimal cache sizes based on system memory
cache_sizes = calculate_optimal_cache_size()
print(f"Using cache sizes: {cache_sizes}")

# Initialize with optimal cache settings
predictor = Predictpy(
    config={'cache_config': cache_sizes}
)
```

The `calculate_optimal_cache_size()` function:
- Detects available system memory (requires `psutil` package)
- Allocates approximately 5% of available memory for caching (capped at 200MB)
- Distributes memory between prediction cache (80%) and completion cache (20%)
- Takes into account estimated entry sizes (predictions ~200 bytes, completions ~2KB)
- Falls back to default values if `psutil` is not installed

### Manual Cache Configuration

You can manually configure cache settings in the config dictionary:

```python
predictor = Predictpy(
    config={
        "cache_config": {
            "predict_size": 5000,    # More predictions for high-query applications
            "completion_size": 50,    # Fewer completions to save memory
            "ttl_seconds": 7200      # 2 hour cache lifetime
        }
    }
)
```

### Cache Monitoring

Monitor cache performance using the `cache_info` property:

```python
# Check cache performance
cache_info = predictor.cache_info
print(f"Cache hit rate: {cache_info['predict_cache']['hit_rate']:.2%}")
print(f"Cache size: {cache_info['predict_cache']['currsize']}/{cache_info['predict_cache']['maxsize']}")
print(f"Modifications since clear: {cache_info['modifications_since_clear']}")
```

### Cache Invalidation

Caches are automatically invalidated when:
- The time-to-live period expires (`ttl_seconds`)
- Too many modifications are recorded (after 50 selections)
- You manually clear the cache with `clear_all_caches()`

```python
# Force clear all caches
predictor.clear_all_caches()
```
