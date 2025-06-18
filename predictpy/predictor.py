"""
Main word prediction module based on n-gram language models.
"""
import os
import sqlite3
from pathlib import Path
import logging
from collections import Counter
import re
import nltk
from datasets import load_dataset
from typing import List, Tuple, Set, Dict, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WordPredictor:
	"""
	Word prediction engine using n-gram language models stored in SQLite.
	
	This class provides word predictions based on context words and partial input,
	with training capabilities built-in.
	"""
	
	def __init__(self, db_path: Optional[str] = None, auto_train: bool = True, target_sentences: int = 10000):
		"""
		Initialize the word predictor with the specified database.
		
		Args:
			db_path: Path to the SQLite database file. If None, uses default location.
			auto_train: Whether to automatically train the model if the database doesn't exist.
			target_sentences: Number of sentences to use for training if auto_train is True.
		"""
		# Set up database path
		if db_path is None:
			# Use package directory to store the database
			self.db_path = os.path.join(os.path.expanduser('~'), '.Predictpy', 'word_predictor.db')
			# Ensure directory exists
			os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
		else:
			self.db_path = db_path
		
		# Check if database exists and train if necessary
		if not os.path.exists(self.db_path):
			if auto_train:
				logging.info(f"Database not found at {self.db_path}, training model...")
				self._setup_nltk()
				self._train_model(target_sentences)
			else:
				raise FileNotFoundError(f"Database not found at {self.db_path} and auto_train is disabled.")
		
		# Connect to database
		self.conn = sqlite3.connect(self.db_path)
		self.conn.row_factory = sqlite3.Row
		logging.info(f"Word predictor initialized with database: {self.db_path}")

	def _setup_nltk(self):
		"""Download necessary NLTK data."""
		for package in ['punkt', 'words']:
			try:
				nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
			except:
				logging.info(f"Downloading NLTK package: {package}")
				nltk.download(package, quiet=True)
	
	def _create_database(self):
		"""Create SQLite database with proper tables and indexes."""
		conn = sqlite3.connect(self.db_path)
		c = conn.cursor()
		
		# Drop existing tables
		c.execute("DROP TABLE IF EXISTS ngrams")
		c.execute("DROP TABLE IF EXISTS words")
		
		# Create tables
		c.execute('''CREATE TABLE words (
			word TEXT PRIMARY KEY,
			frequency INTEGER,
			is_starter BOOLEAN,
			first_letter TEXT,
			first_two TEXT,
			first_three TEXT
		)''')
		
		c.execute('''CREATE TABLE ngrams (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			type TEXT,  -- 'bigram', 'trigram'
			context TEXT,
			word TEXT,
			frequency INTEGER,
			first_letter TEXT
		)''')
		
		# Create indexes for fast lookups
		c.execute("CREATE INDEX idx_words_prefix ON words(first_letter, first_two, first_three)")
		c.execute("CREATE INDEX idx_words_starter ON words(is_starter, frequency DESC)")
		c.execute("CREATE INDEX idx_ngrams_lookup ON ngrams(type, context, first_letter)")
		c.execute("CREATE INDEX idx_ngrams_freq ON ngrams(type, context, frequency DESC)")
		
		conn.commit()
		return conn

	def get_sentence_starters(self, count: int = 10, partial_word: str = "") -> List[str]:
		"""
		Get a list of common sentence starter words.
		
		Args:
			count: Number of starter words to return.
			partial_word: Optional partial word to filter starters.
		
		Returns:
			List of common sentence starter words.
		"""
		c = self.conn.cursor()
		
		if partial_word:
			query = """
				SELECT word FROM words 
				WHERE is_starter = 1 AND word LIKE ?
				ORDER BY frequency DESC
				LIMIT ?
			"""
			c.execute(query, (f"{partial_word}%", count))
		else:
			query = """
				SELECT word FROM words 
				WHERE is_starter = 1
				ORDER BY frequency DESC
				LIMIT ?
			"""
			c.execute(query, (count,))
			
		return [row['word'] for row in c.fetchall()]

	def _train_model(self, target_sentences: int = 10000):
		"""Train the language model using DailyDialog dataset.
		
		Args:
			target_sentences: Number of sentences to use for training
		"""
		logging.info(f"Starting model training with {target_sentences} sentences...")
		
		# Setup NLTK
		self._setup_nltk()
		
		# Load English dictionary
		logging.info("Loading English dictionary...")
		initial_english_words = set(nltk.corpus.words.words())
		english_words = {word.lower() for word in initial_english_words if len(word) > 1 or word.lower() in ['a', 'i']}
		
		# Add common contractions
		common_contractions = {
			"don't", "can't", "won't", "it's", "isn't", "wasn't", "weren't", 
			"he's", "she's", "i'm", "you're", "we're", "they're", 
			"i've", "you've", "we've", "they've", "i'd", "you'd", 
			"he'd", "she'd", "we'd", "they'd", "i'll", "you'll", 
			"he'll", "she'll", "we'll", "they'll", "aren't", "couldn't", 
			"didn't", "doesn't", "hadn't", "hasn't", "haven't", "shouldn't", 
			"wouldn't", "that's", "what's", "where's", "when's", "who's", 
			"why's", "how's", "let's", "there's", "here's"
		}
		english_words.update(common_contractions)
		
		# Remove fragments
		english_words -= {'don', 't', 's'}
		logging.info(f"Dictionary loaded with {len(english_words)} words")
		
		# Create database
		conn = self._create_database()
		c = conn.cursor()
		
		# Collect sentences
		sentences = self._collect_sentences(target_sentences)
		logging.info(f"Collected {len(sentences)} sentences")
		
		# Process sentences
		word_regex = r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b"
		word_counts = Counter()
		bigram_counts = Counter()
		trigram_counts = Counter()
		starter_counts = Counter()
		
		logging.info("Processing sentences...")
		for i, sentence in enumerate(sentences):
			if i % 10000 == 0 and i > 0:
				logging.info(f"Processed {i}/{len(sentences)} sentences")
			
			all_words = re.findall(word_regex, sentence.lower())
			words = [w for w in all_words if w in english_words]
			
			if len(words) < 2:
				continue
			
			# Count starters
			if words[0] in english_words:
				starter_counts[words[0]] += 1
			
			# Count words and n-grams
			for j in range(len(words)):
				word_counts[words[j]] += 1
				
				if j < len(words) - 1:
					bigram = (words[j], words[j+1])
					bigram_counts[bigram] += 1
				
				if j < len(words) - 2:
					trigram = (f"{words[j]} {words[j+1]}", words[j+2])
					trigram_counts[trigram] += 1
		
		# Insert words
		logging.info("Inserting words into database...")
		word_data = []
		for word, count in word_counts.most_common():
			if count < 3:
				break
			
			is_starter = starter_counts.get(word, 0) / count > 0.1
			first_letter = word[0]
			first_two = word[:2] if len(word) >= 2 else None
			first_three = word[:3] if len(word) >= 3 else None
			
			word_data.append((word, count, is_starter, first_letter, first_two, first_three))
		
		c.executemany("INSERT INTO words VALUES (?, ?, ?, ?, ?, ?)", word_data)
		logging.info(f"Inserted {len(word_data)} words")
		
		# Insert bigrams
		logging.info("Inserting bigrams...")
		bigram_data = []
		for (context, word), count in bigram_counts.most_common():
			if count < 2:
				break
			bigram_data.append(('bigram', context, word, count, word[0]))
		
		c.executemany("INSERT INTO ngrams (type, context, word, frequency, first_letter) VALUES (?, ?, ?, ?, ?)", bigram_data)
		logging.info(f"Inserted {len(bigram_data)} bigrams")
		
		# Insert trigrams
		logging.info("Inserting trigrams...")
		trigram_data = []
		for (context, word), count in trigram_counts.most_common():
			if count < 2:
				break
			trigram_data.append(('trigram', context, word, count, word[0]))
		
		c.executemany("INSERT INTO ngrams (type, context, word, frequency, first_letter) VALUES (?, ?, ?, ?, ?)", trigram_data)
		logging.info(f"Inserted {len(trigram_data)} trigrams")
		
		conn.commit()
		conn.close()

	def predict(self, context_words: List[str], partial_word: str = "", max_suggestions: int = 10) -> Tuple[List[str], Dict[str, Any]]:
		"""
		Predict next words based on context and partial word input.
		
		Args:
			context_words: List of words providing context.
			partial_word: Partially typed word to be completed.
			max_suggestions: Maximum number of suggestions to return.
			
		Returns:
			A tuple containing:
			- A list of suggested words.
			- A dictionary with debug information.
		"""
		# ... existing code ...