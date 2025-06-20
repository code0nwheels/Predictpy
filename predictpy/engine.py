"""
Combined word prediction engine that integrates both n-gram models and personal learning.
"""
import os
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from .predictor import WordPredictor
from .personal import PersonalModel
import sqlite3
import threading
from contextlib import contextmanager

class WordPredictionEngine:
    """
    Integrated word prediction engine that combines statistical n-gram predictions
    with personalized predictions based on user history.
    
    This is the main class that should be used for word prediction.
    """
    
    def __init__(self, db_path: Optional[str] = None, auto_train: bool = True,
                target_sentences: int = 10000):
        """
        Initialize the word prediction engine.
        
        Args:
            db_path: Path to the single database file. If None, uses default location.
            auto_train: Whether to automatically train the model if the database doesn't exist.
            target_sentences: Number of sentences to use for training if auto_train is True.
        """
        # Set up database path
        if db_path is None:
            base_dir = os.path.join(os.path.expanduser('~'), '.Predictpy')
            # Create base directory if it doesn't exist
            os.makedirs(base_dir, exist_ok=True)
            db_path = os.path.join(base_dir, 'predictpy.db')
        
        # Create shared connection
        self.shared_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.shared_conn.row_factory = sqlite3.Row
        
        # Add thread lock for database operations
        self._db_lock = threading.Lock()
        
        # Enable WAL mode for better concurrency
        self.shared_conn.execute("PRAGMA journal_mode=WAL")
        self.shared_conn.commit()

        # Initialize both prediction models with the same database file
        self.predictor = WordPredictor(
            db_path=db_path,
            shared_conn=self.shared_conn,  # Add parameter
            auto_train=auto_train,
            target_sentences=target_sentences
        )
        
        self.personal_model = PersonalModel(
            db_path=db_path,
            shared_conn=self.shared_conn  # Add parameter
        )
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe connection access."""
        with self._db_lock:
            yield self.shared_conn

    def predict(self, context_words: List[str], partial_word: str = "",
                max_suggestions: int = 5, learn: bool = True, selected_index: int = -1) -> List[str]:
        """
        Get word predictions based on context and partial input.
        
        Args:
            context_words: List of preceding words that form the context
            partial_word: Partially typed word to complete
            max_suggestions: Maximum number of suggestions to return
            learn: Whether to automatically learn from selections (default: True)
            selected_index: Index of selected prediction, if known (-1 = no selection)
            
        Returns:
            List of predicted words
        """
        # Normalize input
        context_words = [w.lower() for w in context_words if w]
        partial_word = partial_word.lower() if partial_word else ""
        
        # Get personal predictions
        personal_predictions = self.personal_model.get_personal_predictions(
            context_words, partial_word
        )
          # Get base predictions
        base_predictions, _ = self.predictor.predict(context_words, partial_word, max_suggestions)
        
        # Combine predictions, giving priority to personal ones
        results = []
        seen_words = set()
        
        # Add personal predictions first
        for word, _ in personal_predictions:
            if word not in seen_words:
                results.append(word)
                seen_words.add(word)
                
        # Add base predictions
        for word in base_predictions:
            if word not in seen_words:
                results.append(word)
                seen_words.add(word)
                
        # Apply fallback strategy if predictions are empty or insufficient
        if len(results) < max_suggestions:
            # Get most frequent words as fallback
            fallback_words = self._get_top_words(partial_word, max_suggestions - len(results))
            for word in fallback_words:
                if word not in seen_words:
                    results.append(word)
                    seen_words.add(word)
        
        final_predictions = results[:max_suggestions]
        
        # Automatically learn from selection if enabled
        if learn and selected_index >= 0 and selected_index < len(final_predictions):
            selected_word = final_predictions[selected_index]
            self.learn(context_words, selected_word)
            
        return final_predictions

    def get_sentence_starters(self, count: int = 10, partial_word: str = "") -> List[str]:
        """
        Get a list of common sentence starter words.
        
        Args:
            count: Number of starter words to return.
            partial_word: Optional partial word to filter starters.
        
        Returns:
            List of common sentence starter words.
        """
        return self.predictor.get_sentence_starters(count, partial_word)
    
    def learn(self, context_words: List[str], selected_word: str):
        """
        Record user word selection to improve future predictions.
        
        Args:
            context_words: List of preceding words that form the context
            selected_word: The word selected by the user
        """
        self.personal_model.record_selection(context_words, selected_word)
    
    def record_selection(self, context_words: List[str], selected_word: str):
        """
        Record that user selected this word in this context.
        
        Args:
            context_words: List of preceding words that form the context
            selected_word: The word selected by the user
        """
        self.personal_model.record_selection(context_words, selected_word)
    
    def view_personal_data(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the personal data stored in the database.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of selection data dictionaries
        """
        return self.personal_model.view_personal_data(limit)
    
    def get_vocab_count(self) -> int:
        """
        Get the total number of unique words in the vocabulary.

        Returns:
            The total number of unique words.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if the words table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='words'")
                if not cursor.fetchone():
                    # Table doesn't exist, return 0
                    return 0
                
                query = "SELECT COUNT(*) FROM words"
                result = cursor.execute(query).fetchone()
                return result[0] if result else 0
        except sqlite3.Error:
            # Handle any SQLite errors gracefully
            return 0

    def _get_top_words(self, partial_word: str = "", limit: int = 8) -> List[str]:
        """
        Get the most frequent words from the database that match the partial word.
        
        Args:
            partial_word: Optional partial word to filter results
            limit: Maximum number of words to return
            
        Returns:
            List of words sorted by frequency
        """
        cursor = self.predictor.conn.cursor()
        
        if partial_word:
            # If we have a partial word, filter by words starting with that prefix
            if len(partial_word) >= 3:
                query = """
                    SELECT word, frequency 
                    FROM words 
                    WHERE first_three = ?
                    ORDER BY frequency DESC
                    LIMIT ?
                """
                results = cursor.execute(query, (partial_word[:3], limit)).fetchall()
            elif len(partial_word) >= 2:
                query = """
                    SELECT word, frequency 
                    FROM words 
                    WHERE first_two = ?
                    ORDER BY frequency DESC
                    LIMIT ?
                """
                results = cursor.execute(query, (partial_word[:2], limit)).fetchall()
            else:
                query = """
                    SELECT word, frequency 
                    FROM words 
                    WHERE first_letter = ?
                    ORDER BY frequency DESC
                    LIMIT ?
                """
                results = cursor.execute(query, (partial_word[0], limit)).fetchall()
            
            # Further filter for exact prefix match and sort by frequency
            return [r['word'] for r in results if r['word'].startswith(partial_word)][:limit]
        else:
            # If no partial word, just get the most frequent words
            query = """
                SELECT word, frequency 
                FROM words 
                ORDER BY frequency DESC
                LIMIT ?
            """
            results = cursor.execute(query, (limit,)).fetchall()
            return [r['word'] for r in results][:limit]


def main():
    """Main entry point for CLI interface."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Predictpy - Word prediction with personal learning")
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive prediction mode')
    parser.add_argument('--train', '-t', action='store_true', help='Train the model')
    parser.add_argument('--sentences', '-s', type=int, default=10000, help='Number of sentences to use for training')
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training model with {args.sentences} sentences...")
        engine = WordPredictionEngine(auto_train=True, target_sentences=args.sentences)
        print("Training complete!")
        sys.exit(0)
        
    if args.interactive:
        engine = WordPredictionEngine()
        print("Predictpy Interactive Mode")
        print("Enter context followed by a partial word:")
        
        while True:
            try:
                user_input = input("\nInput (or 'q' to quit) > ")
                if user_input.lower() == 'q':
                    break
                    
                words = user_input.split()
                if not words:
                    continue
                    
                # Last word is the partial word, everything before is context
                context = words[:-1]
                partial = words[-1]
                
                print(f"Context: {' '.join(context)}")
                print(f"Partial: {partial}")
                
                # Get predictions
                predictions = engine.predict(context, partial)
                print("\nPredictions:")
                for i, word in enumerate(predictions, 1):
                    print(f"{i}. {word}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    main()
