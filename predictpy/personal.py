"""
Personal word prediction model that learns from user's word selections.
"""
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

class PersonalModel:
    """
    Personal word prediction model that learns from user's word selections.
    
    This class maintains a database of word selections in specific contexts
    to personalize word predictions based on the user's writing patterns.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the personal model with the specified database.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default location.
        """
        # Set up database path
        if db_path is None:
            # Use home directory to store the database
            self.db_path = os.path.join(os.path.expanduser('~'), '.Predictpy', 'personal_model.db')
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        else:
            self.db_path = db_path
        
        self._init_db()
        
    def _init_db(self):
        """Initialize personal model database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS personal_selections (
            context TEXT,
            selected_word TEXT,
            count INTEGER DEFAULT 1,
            last_selected TIMESTAMP,
            PRIMARY KEY (context, selected_word)
        )''')
        
        c.execute('''CREATE INDEX IF NOT EXISTS idx_personal_selections 
                     ON personal_selections(context, count DESC)''')
        
        conn.commit()
        conn.close()
    
    def record_selection(self, context_words: List[str], selected_word: str):
        """
        Record that user selected this word in this context.
        
        Args:
            context_words: List of preceding words that form the context
            selected_word: The word selected by the user
        """        # Use last 2 words as context (or less if not available)
        context = ' '.join(context_words[-2:]) if context_words else 'START'
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        now = datetime.now()
        c.execute('''INSERT INTO personal_selections (context, selected_word, last_selected)
                     VALUES (?, ?, ?)
                     ON CONFLICT(context, selected_word) 
                     DO UPDATE SET count = count + 1, last_selected = ?''',
                  (context, selected_word, now, now))
        
        conn.commit()
        conn.close()
    
    def get_personal_predictions(self, context_words: List[str], partial_word: str = "") -> List[Tuple[str, int]]:
        """
        Get predictions based on personal history.
        
        Args:
            context_words: List of preceding words that form the context
            partial_word: Partially typed word to complete
            
        Returns:
            List of (word, score) tuples, where score is boosted by 1000x
        """
        context = ' '.join(context_words[-2:]) if context_words else 'START'
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        if partial_word:
            results = c.execute('''
                SELECT selected_word, count                FROM personal_selections 
                WHERE context = ? AND selected_word LIKE ?
                ORDER BY count DESC
                LIMIT 5
            ''', (context, partial_word + '%')).fetchall()
        else:            results = c.execute('''
                SELECT selected_word, count 
                FROM personal_selections 
                WHERE context = ?
                ORDER BY count DESC
                LIMIT 5
            ''', (context,)).fetchall()
        
        conn.close()
        
        # Return with boosted frequency (multiply by 1000 to ensure priority)
        return [(r['selected_word'], r['count'] * 1000) for r in results]
    
    def get_personal_sentence_starters(self, partial_word: str = "") -> List[Tuple[str, int]]:
        """
        Get sentence starters based on personal history.
        
        Args:
            partial_word: Partially typed word to complete
            
        Returns:
            List of (word, score) tuples.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        context = 'START'
        
        if partial_word:
            results = c.execute("""
                SELECT selected_word, count FROM personal_selections 
                WHERE context = ? AND selected_word LIKE ?
                ORDER BY count DESC
                LIMIT 10
            """, (context, partial_word + '%')).fetchall()
        else:
            results = c.execute("""
                SELECT selected_word, count 
                FROM personal_selections 
                WHERE context = ?
                ORDER BY count DESC
                LIMIT 10
            """, (context,)).fetchall()
        
        conn.close()
        
        # Return with high score to ensure priority
        return [(r['selected_word'], r['count'] * 1000) for r in results]
    
    def get_boost_factor(self, context_words: List[str], word: str) -> int:
        """
        Get frequency boost for a word in context.
        
        Args:
            context_words: List of preceding words that form the context
            word: The word to check for boosting
            
        Returns:
            Boost factor (2x per selection)
        """
        context = ' '.join(context_words[-2:]) if context_words else 'START'
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        result = c.execute('''            SELECT count FROM personal_selections 
            WHERE context = ? AND selected_word = ?
        ''', (context, word)).fetchone()
        
        conn.close()
        
        # Return 2x boost for each time selected
        return (result[0] * 2) if result else 0
    
    def view_personal_data(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the personal data stored in the database.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of selection data dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        results = c.execute('''            SELECT context, selected_word, count, last_selected
            FROM personal_selections
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        
        conn.close()
        
        return [dict(row) for row in results]
