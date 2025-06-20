"""
English word list manager.
Downloads and caches word lists for use in predictions.
"""
import os
import urllib.request
import sqlite3
import logging
from typing import Optional, Set
import json

class WordList:
    """
    English word list manager.
    Just loads a comprehensive list of valid English words.
    No SpaCy, no frequencies, just words.
    """
    
    # Word list sources
    WORD_LISTS = {
        # 466k words from dwyl/english-words
        'comprehensive': {
            'url': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt',
            'description': '466k English words (alphabetic only)'
        },        # 370k words as JSON (same source, different format)
        'json': {
            'url': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json',
            'description': '370k English words as JSON'
        },
        # Smaller list if you want faster loading
        'common': {
            'url': 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt',
            'description': '10k most common English words'
        }
    }
    
    def __init__(self, list_type: str = 'comprehensive'):
        """
        Initialize word list manager.
        
        Args:
            list_type: 'comprehensive' (466k words), 'json', or 'common' (10k words)
        """
        self.list_type = list_type        # Look for word lists in package data directory first
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(package_dir, 'data')
        
        # Fall back to user cache directory
        self.cache_dir = os.path.join(os.path.expanduser('~'), '.predictpy', 'wordlists')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._words: Optional[Set[str]] = None
    
    def get_wordlist_path(self) -> str:
        """Get path to cached word list."""
        # Define filenames
        if self.list_type == 'comprehensive':
            filename = "words_alpha.txt"  # Match the file we downloaded to data dir
        elif self.list_type == 'common': 
            filename = "common_words.txt"
        elif self.list_type == 'json':
            filename = "words_dictionary.json"
        else:
            filename = f"english_words_{self.list_type}.txt"
              # Check for built-in data file
        package_path = os.path.join(self.data_dir, filename)
        if os.path.exists(package_path):
            return package_path
            
        # Fall back to user cache file
        return os.path.join(self.cache_dir, filename)
    
    def download_if_needed(self) -> str:
        """Get or download word list."""
        filepath = self.get_wordlist_path()
        
        # If filepath points to a file in our package data dir, or if it already exists in user cache
        if os.path.exists(filepath):
            logging.info(f"Using existing word list: {filepath}")
            return filepath
            
        # Need to download to user cache
        user_cache_file = os.path.join(self.cache_dir, os.path.basename(filepath))
        config = self.WORD_LISTS[self.list_type]
        logging.info(f"Downloading {config['description']}...")
        
        try:
            urllib.request.urlretrieve(config['url'], user_cache_file)
            logging.info(f"Downloaded to {user_cache_file}")
            return user_cache_file
        except Exception as e:
            logging.error(f"Failed to download word list: {e}")
            raise
    
    def load_words(self) -> Set[str]:
        """Load words into memory."""
        if self._words is not None:
            return self._words
        
        filepath = self.download_if_needed()
        
        if self.list_type == 'json':
            # Load JSON format
            with open(filepath, 'r', encoding='utf-8') as f:
                words_dict = json.load(f)
                self._words = set(words_dict.keys())
        else:
            # Load text format (one word per line)
            self._words = set()
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and word.isalpha():  # Only alphabetic words
                        self._words.add(word)
        
        logging.info(f"Loaded {len(self._words):,} words")
        return self._words
    
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid English."""
        if self._words is None:
            self.load_words()
        return word.lower() in self._words
    
    def populate_database(self, conn: sqlite3.Connection):
        """Populate database with all words."""
        words = self.load_words()
        cursor = conn.cursor()
        
        # Prepare batch insert data
        word_data = []
        for word in words:
            if 2 <= len(word) <= 20:  # Reasonable word length
                word_lower = word.lower()
                word_data.append((
                    word_lower,
                    100,  # Default frequency (we don't have real frequencies)
                    len(word) <= 2,  # Short words can be starters
                    word_lower[0],
                    word_lower[:2] if len(word_lower) >= 2 else None,
                    word_lower[:3] if len(word_lower) >= 3 else None
                ))
        
        # Insert in batches
        batch_size = 10000
        total = len(word_data)
        logging.info(f"Inserting {total:,} words into database...")
        
        for i in range(0, total, batch_size):
            batch = word_data[i:i + batch_size]
            cursor.executemany(
                "INSERT OR REPLACE INTO words VALUES (?, ?, ?, ?, ?, ?)",
                batch
            )
            if i % 50000 == 0 and i > 0:
                conn.commit()
                logging.info(f"Inserted {i:,} words...")
        
        conn.commit()
        logging.info(f"Successfully inserted {total:,} words")
        return total
