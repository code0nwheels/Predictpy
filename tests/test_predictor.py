"""
Tests for the WordPredictor class
"""
import os
import tempfile
import unittest
from predictpy import WordPredictor

class TestWordPredictor(unittest.TestCase):
    """Test the WordPredictor class."""
    
    def setUp(self):
        """Set up for tests."""
        # Create a temporary database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, 'test_predictor.db')
        # Initialize with very small training set
        self.predictor = WordPredictor(
            db_path=self.db_path, 
            auto_train=True, 
            target_sentences=100
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_predict_empty_context(self):
        """Test prediction with empty context."""
        words, method = self.predictor.predict([], "")
        self.assertIsInstance(words, list)
        self.assertIsInstance(method, str)
    
    def test_predict_with_context(self):
        """Test prediction with context."""
        words, method = self.predictor.predict(["I", "am"], "")
        self.assertIsInstance(words, list)
        self.assertIsInstance(method, str)
    
    def test_predict_with_partial(self):
        """Test prediction with partial word."""
        words, method = self.predictor.predict(["I"], "a")
        self.assertIsInstance(words, list)
        for word in words:
            self.assertTrue(word.startswith("a"))

if __name__ == "__main__":
    unittest.main()
