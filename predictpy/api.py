"""
Enhanced API for Predictpy - Simplified integration interface
"""
from typing import List, Dict, Any, Optional, Union, Callable
from .engine import WordPredictionEngine
import json
import os
import logging
import time
from functools import lru_cache

def _get_version():
    """Get version without circular import."""
    try:
        # Try to read from __init__.py to get the actual version
        init_file = os.path.join(os.path.dirname(__file__), '__init__.py')
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    except Exception:
        pass
    return "0.6.0"  # Update fallback to match your current version

LOW_MEMORY_CONFIG = {
    "training_size": "small",
    "use_semantic": False,
    "target_sentences": 1000,
    "max_personal_selections": 1000,
    "auto_cleanup_days": 14
}

class Predictpy:
    """
    Simple, intuitive API for word prediction.
    
    Quick start:
        predictor = Predictpy()
        suggestions = predictor.predict("I want to")
    """
    
    def __init__(self, 
                 config: Optional[Union[str, Dict[str, Any]]] = None,
                 db_path: Optional[str] = None,
                 auto_train: bool = True,
                 training_size: str = "small",
                 use_semantic: bool = True):
        """
        Initialize Predictpy with minimal configuration.
        
        Args:
            config: Path to config file or config dict
            db_path: Custom database path (optional)
            auto_train: Auto-train if no database exists
            training_size: "small" (1k), "medium" (10k), or "large" (50k) sentences
            use_semantic: Enable semantic completion features (requires ChromaDB)
        """
        # Load config if provided
        if isinstance(config, str) and os.path.exists(config):
            with open(config, 'r') as f:
                config = json.load(f)
        elif config is None:
            config = {}
        
        # Map training sizes
        size_map = {
            "small": 100000,
            "medium": 1000000,
            "large": 5000000
        }
        target_sentences = size_map.get(training_size, 10000)
        
        # Cache configuration
        self._cache_config = config.get('cache_config', {
            'predict_size': 1000,
            'completion_size': 100,
            'ttl_seconds': 3600  # 1 hour
        })
        
        # Track modifications
        self._last_modification = time.time()
        self._modification_count = 0
        
        # Initialize engine with config
        self.engine = WordPredictionEngine(
            db_path=config.get('db_path', db_path),
            auto_train=config.get('auto_train', auto_train),
            target_sentences=config.get('target_sentences', target_sentences)
        )
        
        # Setup caches
        self._setup_caching()
        
        self._semantic = None
        semantic_path = None
        if use_semantic:
            # Set up semantic DB path relative to main DB
            if db_path:
                semantic_path = os.path.join(os.path.dirname(db_path), 'chroma')
            else:
                semantic_path = os.path.join(os.path.expanduser('~'), '.predictpy', 'chroma')
        
        self._semantic_config = {
            'db_path': semantic_path,
            'use_semantic': use_semantic
        }
          # Callbacks for integration
        self._on_prediction_callback = None
        self._on_selection_callback = None
        
    @property
    def semantic(self):
        """Lazy load semantic memory only when accessed."""
        if self._semantic is None and self._semantic_config['use_semantic']:
            try:
                from .semantic import SemanticMemory
                self._semantic = SemanticMemory(db_path=self._semantic_config['db_path'])
            except Exception as e:
                logging.warning(f"Failed to initialize semantic memory: {e}")
                self._semantic = False  # Mark as failed
        return self._semantic if self._semantic is not False else None
        
    def predict(self, text: Union[str, List[str]], count: int = 5) -> List[str]:
        """Cached prediction with smart invalidation."""
        # Check if we should clear stale cache
        self._check_cache_staleness()
        
        if isinstance(text, str):
            words = text.strip().split()
            if text.endswith(' '):
                context, partial = words, ""
            else:
                context, partial = words[:-1], words[-1] if words else ""
        else:
            context, partial = text, ""
        
        # Prepare cache key
        cache_key = (tuple(context), partial, count)
        
        # Get from cache
        return list(self._predict_cache(*cache_key))
    
    def get_vocab_count(self) -> int:
        """
        Get the total number of unique words in the vocabulary.

        Returns:
            The total number of unique words.
        """
        return self.engine.get_vocab_count()

    def get_sentence_starters(self, count: int = 10, partial_word: str = "") -> List[str]:
        """
        Get a list of common sentence starter words.
        
        Args:
            count: Number of starter words to return.
            partial_word: Optional partial word to filter starters.
        
        Returns:
            List of common sentence starter words.
        """
        return self.engine.get_sentence_starters(count, partial_word)
    def select(self, 
               context: Union[str, List[str]], 
               word: str,
               index: Optional[int] = None):
        """
        Record a word selection to improve predictions.
        
        Args:
            context: Context when word was selected
            word: The selected word
            index: Optional index if selecting from predictions
            
        Examples:
            >>> predictor.select("I want", "to")
            >>> predictor.select(["I", "want"], "to", index=0)
        """
        if isinstance(context, str):
            context = context.strip().split()
        
        self.engine.record_selection(context, word)
        
        # Track modification for cache invalidation
        self._modification_count += 1
        self._last_modification = time.time()
        
        # Invalidate cache if too many modifications
        if self._modification_count > 50:
            self._partial_cache_clear()
            self._modification_count = 0
          # Trigger callback if set
        if self._on_selection_callback:
            self._on_selection_callback(context, word, index)
    
    def learn_from_text(self, text: str, text_type: str = "general", tags: Optional[List[str]] = None):
        """
        Learn from a block of text (sentence, paragraph, document).
        
        Args:
            text: Text to learn from
            text_type: Type of text (email, chat, document, etc.)
            tags: Optional tags for categorization
            
        Example:
            >>> predictor.learn_from_text("Hello world. How are you today?", text_type="chat")
        """
        if not text or not text.strip():
            return
        
        # Existing word-level learning
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            words = sentence.strip().lower().split()
            for i in range(len(words)):
                if i > 0:
                    context = words[max(0, i-2):i]
                    self.engine.record_selection(context, words[i])
        
        # New: Store semantic patterns
        if self.semantic:
            try:
                stored_count = self.semantic.store_text(text, text_type=text_type, tags=tags)
                if stored_count > 0:
                    logging.debug(f"Stored {stored_count} semantic patterns")
            except Exception as e:
                logging.warning(f"Failed to store semantic patterns: {e}")
    def predict_completion(self, text: str, min_words: int = 5, 
                          context: Optional[Dict[str, Any]] = None,
                          style: Optional[str] = None,
                          expected_length: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict completion of a thought/paragraph using semantic similarity.
        
        Args:
            text: Partial text to complete
            min_words: Minimum words in completion
            context: Context filters (type, sentiment, formality, etc.)
            style: Style preference (formal, casual, etc.)
            expected_length: Expected completion length (sentence, paragraph)
            
        Returns:
            List of completion suggestions with confidence scores
            
        Examples:
            >>> predictor.predict_completion("I wanted to let you know that")
            >>> predictor.predict_completion("Thanks for your email.", 
                                           context={"type": "email_reply"})
        """
        # Check if we should clear stale cache
        self._check_cache_staleness()
        
        if not hasattr(self, '_semantic') or not self.semantic:
            return []
        
        # Enhance context with style and length preferences
        enhanced_context = context or {}
        if style:
            enhanced_context["style"] = style
        if expected_length:
            enhanced_context["expected_length"] = expected_length
            
        # Use caching if available
        if hasattr(self, '_completion_cache'):
            # Convert context to hashable form
            context_json = json.dumps(enhanced_context, sort_keys=True) if enhanced_context else ""
            
            try:
                # Get from cache
                completions = list(self._completion_cache(text, min_words, context_json))
            except Exception:
                # Fallback to direct call if any error
                completions = self.semantic.predict_completion(
                    text, 
                    n_results=min_words,
                    context=enhanced_context
                )
        else:
            # No cache available
            completions = self.semantic.predict_completion(
                text, 
                n_results=min_words,
                context=enhanced_context
            )
        
        # Filter by minimum words if specified
        filtered = [c for c in completions if c.get('word_count', 0) >= min_words]
        
        return filtered if filtered else completions
        
    def reset_personal_data(self):
        """Clear all personal learning data."""
        conn = self.engine.predictor.conn
        cursor = conn.cursor()
        cursor.execute("DELETE FROM personal_selections")
        conn.commit()
    
    def export_config(self, path: str):
        """Export current configuration."""
        config = {
            'db_path': self.engine.predictor.db_path,
            'version': _get_version()
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Callback support for better integration
    def on_prediction(self, callback: Callable):
        """Set callback for when predictions are made."""
        self._on_prediction_callback = callback
    
    def on_selection(self, callback: Callable):
        """Set callback for when selections are recorded."""
        self._on_selection_callback = callback
    
    # Convenience methods
    def complete(self, partial: str, context: str = "") -> List[str]:
        """
        Complete a partial word.
        
        Args:
            partial: Partial word to complete
            context: Optional context
            
        Returns:
            List of completions
              Example:
            >>> predictor.complete("hel")
            ['hello', 'help', 'helped', 'helping', 'helps']
        """
        context_words = context.strip().split() if context else []
        return self.predict(context_words + [partial])
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = {}
        
        # Personal selections count
        personal_data = self.engine.view_personal_data(limit=10000)
        stats['personal_selections'] = len(personal_data)
        
        # Database size
        if os.path.exists(self.engine.predictor.db_path):
            size_bytes = os.path.getsize(self.engine.predictor.db_path)
            stats['database_size_mb'] = round(size_bytes / (1024 * 1024), 2)
        
        # Top predictions
        if personal_data:
            from collections import Counter
            word_counts = Counter(item['selected_word'] for item in personal_data)
            stats['top_words'] = word_counts.most_common(10)
        
        # Semantic statistics
        if self.semantic:
            try:
                semantic_stats = self.semantic.get_stats()
                stats['semantic'] = semantic_stats
            except Exception as e:
                stats['semantic_error'] = str(e)
        else:
            stats['semantic'] = 'Not available (ChromaDB not installed)'
        
        return stats
    
    def cleanup_semantic_data(self, days: int = 90) -> int:
        """
        Clean up old semantic patterns to free storage space.
        
        Args:
            days: Remove patterns older than this many days
            
        Returns:
            Number of patterns removed
        """
        if not self.semantic:
            return 0
        
        try:
            return self.semantic.cleanup_old_patterns(days)
        except Exception as e:
            logging.error(f"Failed to cleanup semantic data: {e}")
            return 0
    
    def reset_semantic_data(self):
        """Clear all semantic learning data."""
        if self.semantic:
            try:
                # This would require implementing a reset method in SemanticMemory
                # For now, we'll recreate the collection
                self.semantic.collection.delete()
                self.semantic.collection = self.semantic.client.get_or_create_collection(
                    name="user_thoughts",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logging.error(f"Failed to reset semantic data: {e}")
    
    @property
    def has_semantic(self) -> bool:
        """Check if semantic features are available."""
        return self.semantic is not None

    def cleanup_memory(self, aggressive=False):
        """Free up memory immediately."""
        import gc
        
        # Clean personal data
        if aggressive:
            self.reset_personal_data()
        else:
            # Just clean old data
            conn = self.engine.predictor.conn
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM personal_selections 
                WHERE last_selected < datetime('now', '-30 days')
            """)
            conn.commit()
        
        # Clean semantic data
        if self.semantic:
            removed = self.cleanup_semantic_data(days=30 if not aggressive else 7)
            logging.info(f"Removed {removed} semantic patterns")
        
        # Force garbage collection
        gc.collect()
        
        return self.stats  # Return new stats

    def get_memory_usage(self):
        """Get current memory usage of Predictpy components."""
        import sys
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = {
            'total_mb': process.memory_info().rss / 1024 / 1024,
            'components': {},
            'database_sizes': {}
        }
        
        # Check component sizes
        if hasattr(self, 'engine'):
            memory_info['components']['engine'] = sys.getsizeof(self.engine) / 1024 / 1024
        
        if self._semantic is not None:
            memory_info['components']['semantic'] = sys.getsizeof(self._semantic) / 1024 / 1024
        
        # Check database file sizes
        if hasattr(self.engine.predictor, 'db_path'):
            if os.path.exists(self.engine.predictor.db_path):
                db_size = os.path.getsize(self.engine.predictor.db_path) / 1024 / 1024
                memory_info['database_sizes']['main_db'] = f"{db_size:.2f} MB"
        
        # Check semantic database size
        if self._semantic_config.get('db_path'):
            semantic_path = self._semantic_config['db_path']
            if os.path.exists(semantic_path):
                total_size = 0
                for root, dirs, files in os.walk(semantic_path):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))
                memory_info['database_sizes']['semantic_db'] = f"{total_size / 1024 / 1024:.2f} MB"
        
        # Add counts
        try:
            memory_info['counts'] = {
                'vocabulary': self.engine.get_vocab_count(),
                'personal_selections': len(self.engine.view_personal_data(limit=10000))
            }
        except:
            pass
        
        return memory_info

    def _setup_caching(self):
        """Initialize caching with proper configuration."""
        # Word prediction cache (most used)
        self._predict_cache = lru_cache(
            maxsize=self._cache_config.get('predict_size', 1000)
        )(self._predict_uncached)
        
        # Completion cache (memory intensive)
        if hasattr(self, '_semantic') and self._semantic:
            self._completion_cache = lru_cache(
                maxsize=self._cache_config.get('completion_size', 100)
            )(self._completion_uncached)
    
    def _predict_uncached(self, context_tuple: tuple, partial: str, count: int) -> tuple:
        """Actual prediction logic (cached)."""
        context_list = list(context_tuple)
        predictions = self.engine.predict(context_list, partial, count)
        return tuple(predictions)  # Return tuple for caching
    
    def _completion_uncached(self, text: str, min_words: int, context_json: str) -> tuple:
        """Actual completion logic (cached)."""
        if not hasattr(self, '_semantic') or not self._semantic:
            return tuple()
        
        context = json.loads(context_json) if context_json else {}
        completions = self._semantic.predict_completion(text, n_results=min_words, context=context)
        return tuple(completions)
    
    def _check_cache_staleness(self):
        """Clear cache if data is too old."""
        if time.time() - self._last_modification > self._cache_config.get('ttl_seconds', 3600):
            self.clear_all_caches()
    
    def _partial_cache_clear(self):
        """Clear only the most memory-intensive caches."""
        if hasattr(self, '_completion_cache'):
            self._completion_cache.cache_clear()
    
    def clear_all_caches(self):
        """Clear all caches and reset counters."""
        self._predict_cache.cache_clear()
        if hasattr(self, '_completion_cache'):
            self._completion_cache.cache_clear()
        self._modification_count = 0
    
    @property
    def cache_info(self):
        """Get detailed cache statistics."""
        info = {
            'predict_cache': self._predict_cache.cache_info()._asdict(),
            'modifications_since_clear': self._modification_count,
            'last_modification': self._last_modification
        }
        
        if hasattr(self, '_completion_cache'):
            info['completion_cache'] = self._completion_cache.cache_info()._asdict()
        
        # Calculate hit rates
        for cache_name, cache_info in info.items():
            if isinstance(cache_info, dict) and 'hits' in cache_info:
                total = cache_info['hits'] + cache_info['misses']
                cache_info['hit_rate'] = cache_info['hits'] / total if total > 0 else 0
        
        return info
    
    def __del__(self):
        """Clean up resources on deletion."""
        try:
            if hasattr(self, 'engine') and hasattr(self.engine.predictor, 'conn'):
                self.engine.predictor.conn.close()
            
            # Clear semantic model
            if hasattr(self, '_semantic') and self._semantic:
                del self._semantic
                
        except Exception:
            pass  # Ignore errors during cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_memory(aggressive=False)
        return False

# Convenience function for quick usage
def create_predictor(**kwargs) -> Predictpy:
    """Create a Predictpy instance with optional configuration."""
    return Predictpy(**kwargs)