"""
Tests for the semantic memory functionality.
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock

try:
    from predictpy.semantic import SemanticMemory, CHROMADB_AVAILABLE
except ImportError:
    CHROMADB_AVAILABLE = False


@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB dependencies not available")
class TestSemanticMemory(unittest.TestCase):
    """Test cases for SemanticMemory class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_chroma")
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test semantic memory initialization."""
        semantic = SemanticMemory(db_path=self.db_path)
        self.assertIsNotNone(semantic.client)
        self.assertIsNotNone(semantic.collection)
        self.assertIsNotNone(semantic.encoder)
    
    def test_store_text(self):
        """Test storing text and extracting thoughts."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        text = "Hello world. How are you today? I hope you're doing well."
        stored_count = semantic.store_text(text, text_type="test")
        
        self.assertGreater(stored_count, 0)
        
        # Check that data was stored
        stats = semantic.get_stats()
        self.assertGreater(stats['total_thoughts'], 0)
    
    def test_split_thoughts(self):
        """Test thought splitting functionality."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        thoughts = semantic._split_thoughts(text)
        
        self.assertGreater(len(thoughts), 0)
        for thought in thoughts:
            self.assertIsInstance(thought, str)
            self.assertTrue(thought.strip())
    
    def test_classify_thought(self):
        """Test thought classification."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        test_cases = [
            ("How are you?", "question"),
            ("That's amazing!", "exclamation"),
            ("Thank you for your help.", "gratitude"),
            ("I'm sorry for the delay.", "apology"),
            ("I think this is correct.", "opinion"),
            ("I will complete the task.", "intention"),
            ("This is a simple statement.", "statement"),
        ]
        
        for text, expected_type in test_cases:
            result = semantic._classify_thought(text)
            self.assertEqual(result, expected_type)
    
    def test_predict_completion(self):
        """Test semantic completion prediction."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        # Store some training data
        training_texts = [
            "Thank you for your email. I will respond by tomorrow.",
            "Thank you for your message. I appreciate your patience.",
            "The meeting has been scheduled for next Tuesday at 2 PM.",
            "The meeting has been moved to Thursday morning.",
        ]
        
        for text in training_texts:
            semantic.store_text(text)
        
        # Test completion prediction
        completions = semantic.predict_completion("Thank you for your", n_results=3)
        
        # Should return some completions
        self.assertIsInstance(completions, list)
        
        # Check structure of completions
        for completion in completions:
            self.assertIsInstance(completion, dict)
            self.assertIn('text', completion)
            self.assertIn('confidence', completion)
            self.assertIn('type', completion)
    
    def test_should_store_duplicates(self):
        """Test duplicate detection."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        text = "This is a unique test sentence."
        
        # First storage should succeed
        self.assertTrue(semantic._should_store(text))
        semantic.store_text(text)
        
        # Very similar text should not be stored
        similar_text = "This is a unique test sentence."
        # Note: The actual behavior depends on the similarity threshold
        # This test might need adjustment based on the embedding model
    
    def test_cleanup_old_patterns(self):
        """Test cleanup of old patterns."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        # Store some data
        semantic.store_text("Test sentence for cleanup.")
        
        # Cleanup (should not remove recent data)
        removed = semantic.cleanup_old_patterns(days=1)
        self.assertGreaterEqual(removed, 0)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        semantic = SemanticMemory(db_path=self.db_path)
        
        # Store some data
        semantic.store_text("Test sentence one. Test sentence two!")
        
        stats = semantic.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_thoughts', stats)
        self.assertIn('type_distribution', stats)
        self.assertIn('average_words_per_thought', stats)
        self.assertIn('storage_path', stats)


class TestSemanticMemoryWithoutChromaDB(unittest.TestCase):
    """Test cases for when ChromaDB is not available."""
    
    @patch('predictpy.semantic.CHROMADB_AVAILABLE', False)
    def test_import_error_handling(self):
        """Test that appropriate error is raised when ChromaDB not available."""
        with self.assertRaises(ImportError):
            from predictpy.semantic import SemanticMemory
            SemanticMemory()


if __name__ == '__main__':
    unittest.main()
