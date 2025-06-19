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
    
    def __init__(self, db_path: Optional[str] = None, shared_conn: Optional[sqlite3.Connection] = None):
        """
        Initialize the personal model with the specified database.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default location.
            shared_conn: An existing database connection to reuse.
        """
        # Set up database path
        if db_path is None:
            self.db_path = os.path.join(os.path.expanduser('~'), '.Predictpy', 'personal_model.db')
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        else:
            self.db_path = db_path
        
        # Use shared connection if provided
        if shared_conn:
            self.conn = shared_conn
            self.shared = True
        else:
            self.conn = None
            self.shared = False
        
        self._init_db()
        
    def _init_db(self):
        """Initialize personal model database."""
        # Use existing connection or create new one
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        c = self.conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS personal_selections (
            context TEXT,
            selected_word TEXT,
            count INTEGER DEFAULT 1,
            last_selected TIMESTAMP,
            PRIMARY KEY (context, selected_word)
        )''')
        
        c.execute('''CREATE INDEX IF NOT EXISTS idx_personal_selections 
                     ON personal_selections(context, count DESC)''')
        
        self.conn.commit()
        
        # Don't close if using shared connection
        if not self.shared:
            self.conn.close()
            self.conn = None
    
    def record_selection(self, context_words: List[str], selected_word: str):
        """
        Record that user selected this word in this context.
        
        Args:
            context_words: List of preceding words that form the context
            selected_word: The word selected by the user
        """        # Use last 2 words as context (or less if not available)
        context = ' '.join(context_words[-2:]) if context_words else 'START'
        
        # Open connection if needed
        if self.conn is None:
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
        
        c = conn.cursor()

        # ADD: Check and cleanup if too many entries
        count = c.execute("SELECT COUNT(*) FROM personal_selections").fetchone()[0]
        if count > 10000:  # Limit to 10k entries
            # Delete oldest 1000 entries
            c.execute("""
                DELETE FROM personal_selections 
                WHERE rowid IN (
                    SELECT rowid FROM personal_selections 
                    ORDER BY last_selected ASC 
                    LIMIT 1000
                )
            """)
        
        now = datetime.now()
        c.execute('''INSERT INTO personal_selections (context, selected_word, last_selected)
                     VALUES (?, ?, ?)
                     ON CONFLICT(context, selected_word) 
                     DO UPDATE SET count = count + 1, last_selected = ?''',
                  (context, selected_word, now, now))
        
        conn.commit()
        
        # Only close if we opened it
        if self.conn is None:
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
        
        if self.conn is None:
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
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
        
        if self.conn is None:
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
        if self.conn is None:
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
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
        
        if self.conn is None:
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
        
        if self.conn is None:
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
        c = conn.cursor()
        
        result = c.execute('''            SELECT count FROM personal_selections 
            WHERE context = ? AND selected_word = ?
        ''', (context, word)).fetchone()
        
        if self.conn is None:
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
        if self.conn is None:
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        results = c.execute('''            SELECT context, selected_word, count, last_selected
            FROM personal_selections
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        
        if self.conn is None:
            conn.close()
        
        return [dict(row) for row in results]
