"""
Enhanced API for Predictpy - Simplified integration interface
"""
from typing import List, Dict, Any, Optional, Union, Callable
from .engine import WordPredictionEngine
from .semantic import SemanticMemory
import json
import os
import logging

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
    return "0.3.0"  # fallback version

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
        
        # Initialize engine with config
        self.engine = WordPredictionEngine(
            db_path=config.get('db_path', db_path),
            auto_train=config.get('auto_train', auto_train),
            target_sentences=config.get('target_sentences', target_sentences)
        )
          # Initialize semantic memory
        self.semantic = None
        if use_semantic:
            try:
                # Set up semantic DB path relative to main DB
                if db_path:
                    semantic_path = os.path.join(os.path.dirname(db_path), 'chroma')
                else:
                    semantic_path = os.path.join(os.path.expanduser('~'), '.predictpy', 'chroma')
                
                self.semantic = SemanticMemory(db_path=semantic_path)
            except Exception as e:
                logging.warning(f"Failed to initialize semantic memory: {e}")
                self.semantic = None
        
        # Callbacks for integration
        self._on_prediction_callback = None
        self._on_selection_callback = None
    
    def predict(self, text: Union[str, List[str]], count: int = 5) -> List[str]:
        if isinstance(text, str):
            words = text.strip().split()
            if text.endswith(' '):
                context, partial = words, ""
            else:
                context, partial = words[:-1], words[-1] if words else ""
        else:
            context, partial = text, ""
        
        return self.engine.predict(context, partial, count)
    
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
        if not self.semantic:
            return []
        
        # Enhance context with style and length preferences
        enhanced_context = context or {}
        if style:
            enhanced_context["style"] = style
        if expected_length:
            enhanced_context["expected_length"] = expected_length
        
        try:
            completions = self.semantic.predict_completion(
                text, 
                n_results=min_words,
                context=enhanced_context
            )
            
            # Filter by minimum words if specified
            filtered = [c for c in completions if c.get('word_count', 0) >= min_words]
            
            return filtered if filtered else completions
            
        except Exception as e:
            logging.error(f"Failed to predict completion: {e}")
            return []
        
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

# Convenience function for quick usage
def create_predictor(**kwargs) -> Predictpy:
    """Create a Predictpy instance with optional configuration."""
    return Predictpy(**kwargs)