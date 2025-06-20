# Troubleshooting

Common issues and solutions for Predictpy users.

## Installation Issues

### ChromaDB Installation Problems

**Problem:** ChromaDB fails to install or import

```
ImportError: No module named 'chromadb'
```

**Solutions:**

1. **Manual Installation:**
   ```bash
   pip install chromadb sentence-transformers
   ```

2. **Update pip and try again:**
   ```bash
   pip install --upgrade pip
   pip install predictpy
   ```

3. **Use without semantic features:**
   ```python
   predictor = Predictpy(use_semantic=False)
   ```

4. **Platform-specific issues:**
   ```bash
   # On Windows with Python 3.11+
   pip install --upgrade setuptools wheel
   pip install chromadb
   
   # On macOS with M1/M2
   pip install chromadb --no-deps
   pip install sentence-transformers
   ```

### Word List Download Issues

**Problem:** Word list download fails

```
Failed to download word list: <error message>
```

**Solutions:**

1. **Check your internet connection**

2. **Download word lists manually:**
   Download the appropriate word list file directly from GitHub:
   
   - Comprehensive list: https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
   - Common words: https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt
   
   Save the file to `~/.predictpy/wordlists/` with appropriate naming.

3. **Set custom cache directory:**
   You can modify the cache directory in the WordList class if needed.

### Dependencies Conflict

**Problem:** Version conflicts with existing packages

**Solutions:**

1. **Create virtual environment:**
   ```bash
   python -m venv predictpy_env
   # Windows
   predictpy_env\Scripts\activate
   # Unix/macOS
   source predictpy_env/bin/activate
   
   pip install predictpy
   ```

2. **Update conflicting packages:**
   ```bash
   pip install --upgrade datasets
   ```

---

## Runtime Issues

### Database Problems

**Problem:** Database corruption or access issues

```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**

1. **Reset database:**
   ```python
   import os
   from predictpy import Predictpy
   
   # Remove corrupted database
   db_path = os.path.expanduser('~/.predictpy/predictpy.db')
   if os.path.exists(db_path):
       os.remove(db_path)
   
   # Reinitialize
   predictor = Predictpy(auto_train=True)
   ```

2. **Use custom database path:**
   ```python
   predictor = Predictpy(db_path="/tmp/test_predictions.db")
   ```

3. **Check disk space:**
   ```python
   import shutil
   
   free_space = shutil.disk_usage('.').free
   print(f"Free disk space: {free_space / (1024**3):.1f} GB")
   ```

### Memory Issues

**Problem:** High memory usage or out-of-memory errors

**Solutions:**

1. **Disable semantic features:**
   ```python
   predictor = Predictpy(use_semantic=False)
   ```

2. **Use smaller training size:**
   ```python
   predictor = Predictpy(training_size="small")
   ```

3. **Clean up old data:**
   ```python
   # Clean semantic data
   removed = predictor.cleanup_semantic_data(days=30)
   print(f"Removed {removed} old patterns")
   
   # Reset personal data
   predictor.reset_personal_data()
   ```

4. **Monitor memory usage:**
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Memory usage: {memory_mb:.1f} MB")
   ```

### Performance Issues

**Problem:** Slow predictions or initialization

**Solutions:**

1. **Optimize configuration:**
   ```python
   fast_config = {
       "training_size": "small",
       "use_semantic": False,
       "auto_train": False
   }
   predictor = Predictpy(config=fast_config)
   ```

2. **Use caching:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_predict(text):
       return tuple(predictor.predict(text))
   ```

3. **Warm up semantic model:**
   ```python
   if predictor.has_semantic:
       # First prediction is slow, subsequent ones are fast
       predictor.predict_completion("warm up", min_words=1)
   ```

### Cache-Related Problems

**Problem:** Memory usage is too high with caching enabled

**Solutions:**

1. **Use memory-aware cache sizing:**
   ```python
   from predictpy import calculate_optimal_cache_size, Predictpy
   
   # Get system-appropriate cache sizes
   cache_sizes = calculate_optimal_cache_size() 
   
   # Reduce further if needed
   cache_sizes['predict_size'] = int(cache_sizes['predict_size'] * 0.5)
   cache_sizes['completion_size'] = int(cache_sizes['completion_size'] * 0.5)
   
   predictor = Predictpy(config={'cache_config': cache_sizes})
   ```

2. **Disable the completion cache entirely:**
   ```python
   predictor = Predictpy(
       config={
           'cache_config': {
               'predict_size': 1000,
               'completion_size': 0  # Disable completion caching
           }
       }
   )
   ```

**Problem:** Cache doesn't seem to be working (no performance improvement)

**Solutions:**

1. **Check cache hit rate:**
   ```python
   cache_info = predictor.cache_info
   print(f"Cache hit rate: {cache_info['predict_cache']['hit_rate']:.2%}")
   ```

2. **Ensure cache isn't being invalidated too frequently:**
   ```python
   print(f"Modifications since clear: {predictor.cache_info['modifications_since_clear']}")
   print(f"Last modification: {predictor.cache_info['last_modification']}")
   ```

3. **Increase cache TTL:**
   ```python
   predictor = Predictpy(
       config={
           'cache_config': {
               'ttl_seconds': 86400  # 24 hours
           }
       }
   )
   ```

**Problem:** Installing psutil fails for memory-aware caching

**Solution:**

1. **Manual psutil installation:**
   ```bash
   # Windows
   pip install --upgrade setuptools
   pip install psutil
   
   # Linux
   sudo apt-get install python3-dev
   pip install psutil
   
   # macOS
   pip install psutil
   ```

2. **Use Predictpy without psutil:**
   ```python
   # Will use default cache sizes if psutil isn't available
   predictor = Predictpy()
   ```

---

## Semantic Feature Issues

### ChromaDB Connection Problems

**Problem:** ChromaDB fails to initialize

```
chromadb.errors.ChromaError: Could not connect to ChromaDB
```

**Solutions:**

1. **Check ChromaDB installation:**
   ```python
   try:
       import chromadb
       print(f"ChromaDB version: {chromadb.__version__}")
   except ImportError:
       print("ChromaDB not installed")
   ```

2. **Use custom ChromaDB path:**
   ```python
   import tempfile
   temp_dir = tempfile.mkdtemp()
   predictor = Predictpy(
       use_semantic=True,
       config={"semantic_db_path": temp_dir}
   )
   ```

3. **Reset ChromaDB collection:**
   ```python
   predictor.reset_semantic_data()
   ```

### Embedding Model Issues

**Problem:** Sentence transformer model fails to load

**Solutions:**

1. **Use smaller model:**
   ```python
   config = {"semantic_model": "all-MiniLM-L6-v2"}  # Lighter model
   predictor = Predictpy(config=config, use_semantic=True)
   ```

2. **Download model manually:**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')  # Downloads if needed
   ```

3. **Check internet connection:**
   ```python
   import urllib.request
   try:
       urllib.request.urlopen('https://huggingface.co', timeout=5)
       print("Internet connection OK")
   except:
       print("No internet connection for model download")
   ```

### Semantic Completion Quality Issues

**Problem:** Poor or irrelevant completions

**Solutions:**

1. **Provide more training data:**
   ```python
   training_texts = [
       "High-quality example text 1...",
       "High-quality example text 2...",
       "High-quality example text 3..."
   ]
   
   for text in training_texts:
       predictor.learn_from_text(text, text_type="quality_training")
   ```

2. **Use specific context:**
   ```python
   completions = predictor.predict_completion(
       "Thank you for your",
       context={
           "text_type": "email",
           "formality": "business",
           "purpose": "response"
       }
   )
   ```

3. **Check training data quality:**
   ```python
   stats = predictor.stats
   if stats.get('semantic', {}).get('total_patterns', 0) < 10:
       print("Need more training data for better completions")
   ```

---

## Platform-Specific Issues

### Windows Issues

**Problem:** Path-related errors on Windows

**Solutions:**

1. **Use raw strings for paths:**
   ```python
   predictor = Predictpy(db_path=r"C:\Users\Username\predictions.db")
   ```

2. **Handle Unicode paths:**
   ```python
   import os
   db_path = os.path.expanduser('~\\predictpy\\predictions.db')
   predictor = Predictpy(db_path=db_path)
   ```

### macOS Issues

**Problem:** Permission denied errors

**Solutions:**

1. **Check file permissions:**
   ```bash
   ls -la ~/.predictpy/
   chmod 755 ~/.predictpy/
   ```

2. **Use user directory:**
   ```python
   import os
   user_dir = os.path.expanduser('~/Documents/predictpy')
   os.makedirs(user_dir, exist_ok=True)
   predictor = Predictpy(db_path=f"{user_dir}/predictions.db")
   ```

### Linux Issues

**Problem:** Missing system dependencies

**Solutions:**

1. **Install system packages:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   
   # CentOS/RHEL
   sudo yum install python3-devel gcc
   ```

2. **Use conda environment:**
   ```bash
   conda create -n predictpy python=3.9
   conda activate predictpy
   pip install predictpy
   ```

---

## Debugging Tools

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('predictpy')

# Create predictor with debug info
predictor = Predictpy()

# Test with logging
predictions = predictor.predict("hello")
logger.debug(f"Predictions: {predictions}")
```

### Diagnostic Information

```python
def diagnose_predictpy():
    """Comprehensive diagnostic information."""
    import sys
    import platform
    import sqlite3
    
    print("=== Predictpy Diagnostic Information ===")
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Package versions
    try:
        import predictpy
        print(f"Predictpy version: {predictpy.__version__}")
    except:
        print("Predictpy: Not installed")
    
    try:
        import chromadb
        print(f"ChromaDB version: {chromadb.__version__}")
    except:
        print("ChromaDB: Not available")
    
    try:
        import sentence_transformers
        print(f"Sentence-transformers: {sentence_transformers.__version__}")
    except:
        print("Sentence-transformers: Not available")      try:
        from predictpy.wordlist import WordList
        wordlist = WordList()
        print(f"WordList available: True")
    except:
        print("WordList: Not available")
    
    # Database info
    print(f"SQLite version: {sqlite3.sqlite_version}")
    
    # Test basic functionality
    try:
        predictor = Predictpy(use_semantic=False)
        test_predictions = predictor.predict("hello")
        print(f"Basic prediction test: ✓ ({len(test_predictions)} predictions)")
    except Exception as e:
        print(f"Basic prediction test: ✗ ({e})")
    
    # Test semantic functionality
    try:
        predictor = Predictpy(use_semantic=True)
        if predictor.has_semantic:
            print("Semantic features test: ✓")
        else:
            print("Semantic features test: ✗ (Not available)")
    except Exception as e:
        print(f"Semantic features test: ✗ ({e})")

# Run diagnostics
diagnose_predictpy()
```

### Performance Profiling

```python
import time
import cProfile
import pstats

def profile_predictions():
    """Profile prediction performance."""
    
    def test_predictions():
        predictor = Predictpy()
        test_phrases = ["hello", "how are", "thank you"] * 10
        
        for phrase in test_phrases:
            predictions = predictor.predict(phrase)
    
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    test_predictions()
    end_time = time.time()
    
    profiler.disable()
    
    # Print results
    print(f"Total time: {end_time - start_time:.3f} seconds")
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

# Run profiling
# profile_predictions()
```

---

## Getting Help

### Check Status and Statistics

```python
def check_predictor_health(predictor):
    """Check predictor health and status."""
    
    print("=== Predictor Health Check ===")
    
    # Basic functionality
    try:
        test_pred = predictor.predict("test")
        print(f"✓ Basic prediction: {len(test_pred)} results")
    except Exception as e:
        print(f"✗ Basic prediction failed: {e}")
    
    # Semantic functionality
    if predictor.has_semantic:
        try:
            test_comp = predictor.predict_completion("test")
            print(f"✓ Semantic completion: {len(test_comp)} results")
        except Exception as e:
            print(f"✗ Semantic completion failed: {e}")
    else:
        print("- Semantic features not available")
    
    # Statistics
    try:
        stats = predictor.stats
        print(f"✓ Database size: {stats.get('database_size_mb', 0)} MB")
        print(f"✓ Personal selections: {stats.get('personal_selections', 0)}")
        
        semantic_stats = stats.get('semantic')
        if semantic_stats and isinstance(semantic_stats, dict):
            print(f"✓ Semantic patterns: {semantic_stats.get('total_patterns', 0)}")
    except Exception as e:
        print(f"✗ Statistics failed: {e}")

# Usage
predictor = Predictpy()
check_predictor_health(predictor)
```

### Report Issues

When reporting issues, include:

1. **System information:** Run `diagnose_predictpy()` function above
2. **Error messages:** Full traceback if available
3. **Minimal reproduction case:** Simplest code that shows the problem
4. **Expected vs actual behavior:** What should happen vs what does happen

### Community Resources

- **GitHub Issues:** [Report bugs and feature requests](https://github.com/code0nwheels/Predictpy/issues)
- **Documentation:** Check the [full documentation](index.md) for details
- **Examples:** Review [examples](examples.md) for usage patterns

### Common Error Messages and Solutions

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `ImportError: No module named 'chromadb'` | ChromaDB not installed | `pip install chromadb` or use `use_semantic=False` |
| `urllib.error.URLError` | Failed to download word list | Check internet connection or download files manually |
| `sqlite3.DatabaseError: database disk image is malformed` | Corrupted database | Delete database file and reinitialize |
| `MemoryError` | Insufficient memory | Use smaller training size or disable semantic features |
| `PermissionError` | File access denied | Check file permissions or use different path |
| `ConnectionError` | Network issues | Check internet connection for model downloads |
