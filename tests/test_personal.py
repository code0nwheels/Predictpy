"""
Tests for the PersonalModel class
"""
import os
import tempfile
import unittest
from predictpy import PersonalModel

class TestPersonalModel(unittest.TestCase):
    """Test the PersonalModel class."""
    
    def setUp(self):
        """Set up for tests."""
        # Create a temporary database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, 'test_personal.db')
        self.model = PersonalModel(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_record_and_predict(self):
        """Test recording selections and getting predictions."""
        # Record some selections
        self.model.record_selection(["I", "am"], "working")
        self.model.record_selection(["I", "am"], "learning")
        self.model.record_selection(["I", "am"], "working")  # Record twice
        
        # Get predictions
        predictions = self.model.get_personal_predictions(["I", "am"])
        
        # Check results
        self.assertTrue(len(predictions) > 0)
        # 'working' should be first as it was selected twice
        self.assertEqual(predictions[0][0], "working")
    
    def test_boost_factor(self):
        """Test the boost factor calculation."""
        # Record a selection
        self.model.record_selection(["the", "quick"], "brown")
        
        # Check boost factor
        boost = self.model.get_boost_factor(["the", "quick"], "brown")
        self.assertEqual(boost, 2)  # 2x for first selection
        
        # Record another selection of the same word
        self.model.record_selection(["the", "quick"], "brown")
        
        # Check updated boost factor
        boost = self.model.get_boost_factor(["the", "quick"], "brown")
        self.assertEqual(boost, 4)  # 2x for each selection (2 total)
    
    def test_view_data(self):
        """Test viewing personal data."""
        # Record some selections
        self.model.record_selection(["hello"], "world")
        self.model.record_selection(["hello"], "there")
        
        # Get data
        data = self.model.view_personal_data()
        
        # Check results
        self.assertEqual(len(data), 2)
        
        # Words should be in data
        words = [entry["selected_word"] for entry in data]
        self.assertIn("world", words)
        self.assertIn("there", words)

if __name__ == "__main__":
    unittest.main()
