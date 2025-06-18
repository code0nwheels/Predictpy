"""
Tests for the integrated WordPredictionEngine
"""
import os
import tempfile
import unittest
from predictpy import WordPredictionEngine

class TestWordPredictionEngine(unittest.TestCase):
    """Test the WordPredictionEngine class."""
    
    def setUp(self):
        """Set up for tests."""        # Create a temporary directory for the database
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.temp_dir.name, 'predictpy_test.db')
        # Initialize with very small training set
        self.engine = WordPredictionEngine(
            db_path=db_path, 
            auto_train=True, 
            target_sentences=100
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_prediction(self):
        """Test basic prediction functionality."""
        # Get predictions
        predictions = self.engine.predict(["I", "am"])
        self.assertIsInstance(predictions, list)
        self.assertTrue(len(predictions) > 0)
    
    def test_personal_learning(self):
        """Test that personal selections affect predictions."""
        # Get initial predictions
        initial_predictions = self.engine.predict(["I", "am"])
        
        # Record a selection that likely isn't in the initial predictions
        unusual_word = "xylophonist"  # Unlikely to be in top predictions naturally
        self.engine.record_selection(["I", "am"], unusual_word)
        
        # Get predictions after recording
        new_predictions = self.engine.predict(["I", "am"])
        
        # The unusual word should now be in predictions
        self.assertIn(unusual_word, new_predictions)
        
        # And it should be first (highest priority)
        self.assertEqual(new_predictions[0], unusual_word)
    
    def test_auto_learning(self):
        """Test the auto-learning feature."""
        context = ["hello", "world"]
        
        # Get initial predictions
        initial_predictions = self.engine.predict(context)
        
        # Select the first prediction with auto-learning enabled
        selected_index = 0
        if initial_predictions:
            selected_word = initial_predictions[selected_index]
            
            # Call predict with selected_index to trigger auto-learning
            self.engine.predict(context, selected_index=selected_index)
            
            # Get new predictions and verify the selected word is prioritized
            new_predictions = self.engine.predict(context)
            self.assertEqual(new_predictions[0], selected_word)
    
    def test_learn_flag(self):
        """Test that the learn flag controls auto-learning behavior."""
        context = ["test", "learn", "flag"]
        
        # Get initial predictions
        initial_predictions = self.engine.predict(context)
        
        # Select the first prediction with auto-learning disabled
        if initial_predictions:
            selected_index = 0
            selected_word = initial_predictions[selected_index]
            
            # Call predict with learn=False
            self.engine.predict(context, selected_index=selected_index, learn=False)
            
            # Verify data wasn't recorded by checking the personal data
            personal_data = self.engine.view_personal_data()
            context_str = ' '.join(context[-2:])  # Personal model uses last 2 words
            
            # Find if this selection was recorded
            recorded = False
            for entry in personal_data:
                if entry['context'] == context_str and entry['selected_word'] == selected_word:
                    recorded = True
                    break
            
            # Verify it wasn't recorded
            self.assertFalse(recorded)
    
    def test_partial_word(self):
        """Test prediction with partial word."""
        # Record a selection with a distinctive prefix
        self.engine.record_selection(["the"], "zebra")
        
        # Should get "zebra" when looking for words starting with "z"
        predictions = self.engine.predict(["the"], "z")
        self.assertIn("zebra", predictions)

if __name__ == "__main__":
    unittest.main()
