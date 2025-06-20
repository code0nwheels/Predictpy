"""
Main word prediction module based on n-gram language models.
"""
import os
import sys
import sqlite3
from pathlib import Path
import logging
from collections import Counter
import re
from datasets import load_dataset
from typing import List, Tuple, Set, Dict, Optional, Any, Union
from functools import lru_cache

# Import our word list manager
from .wordlist import WordList

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WordPredictor:
	"""
	Word prediction engine using n-gram language models stored in SQLite.
	
	This class provides word predictions based on context words and partial input,
	with training capabilities built-in.
	"""
	
	def __init__(self, db_path: Optional[str] = None, shared_conn: Optional[sqlite3.Connection] = None, 
             auto_train: bool = True, target_sentences: int = 10000):
		"""
		Initialize the word predictor with the specified database.
		
		Args:
			db_path: Path to the SQLite database file. If None, uses default location.
			shared_conn: An existing database connection to reuse.
			auto_train: Whether to automatically train the model if the database doesn't exist.
			target_sentences: Number of sentences to use for training if auto_train is True.
		"""
		# Set up database path
		if db_path is None:
			self.db_path = os.path.join(os.path.expanduser('~'), '.Predictpy', 'word_predictor.db')
			os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
		else:
			self.db_path = db_path
				# Check if database exists and train if necessary
		db_exists = os.path.exists(self.db_path)
		
		# Handle shared connection
		if shared_conn:
			self.conn = shared_conn
			self.conn.row_factory = sqlite3.Row
		# Handle non-existing database
		elif not db_exists:
			if auto_train:
				logging.info(f"Database not found at {self.db_path}, training model...")
				self._setup_wordlist()
				self._train_model(target_sentences)
				# Connection is created in _train_model
				return  # _train_model already created everything we need
			else:
				raise FileNotFoundError(f"Database not found at {self.db_path} and auto_train is disabled.")
		# Handle existing database
		else:
			self.conn = sqlite3.connect(self.db_path)
			self.conn.row_factory = sqlite3.Row
		
		# At this point self.conn is initialized, check if tables exist
		cursor = self.conn.cursor()
		cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('words', 'ngrams')")
		tables = [row[0] for row in cursor.fetchall()]
		tables_exist = len(tables) == 2
		
		if not tables_exist:
			if auto_train:
				logging.info(f"Tables not found in database {self.db_path}, training model...")
				self._setup_wordlist()
				self._train_model(target_sentences)
			else:
				raise FileNotFoundError("Required tables not found in database and auto_train is disabled.")
		elif shared_conn:
			# Use shared connection if provided
			self.conn = shared_conn
		else:
			# Create new connection
			self.conn = sqlite3.connect(self.db_path)
			self.conn.row_factory = sqlite3.Row
		
		# Cache frequent queries
		self._cached_ngram_query = lru_cache(maxsize=2000)(self._query_ngrams)
		self._cached_word_query = lru_cache(maxsize=1000)(self._query_words)
		logging.info(f"Word predictor initialized with database: {self.db_path}")
		
	def _setup_wordlist(self):
		"""Initialize WordList for vocabulary access."""
		try:
			# Create word list with comprehensive English vocabulary
			self.wordlist = WordList('comprehensive')
			logging.info("WordList initialized successfully")
		except Exception as e:
			logging.error(f"Error initializing WordList: {e}")
			raise
	
	def _create_database(self):
		"""Create SQLite database with proper tables and indexes."""
		conn = sqlite3.connect(self.db_path)
		conn.row_factory = sqlite3.Row
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
		
	def _import_wordlist(self):
		"""Import all words from WordList into database."""
		# The WordList class already has a populate_database method
		# that inserts words into the database with the correct schema
		imported = self.wordlist.populate_database(self.conn)
		logging.info(f"Imported {imported} words from WordList")
		return imported

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
		
		# Create database with necessary tables
		self.conn = self._create_database()
		
		# Load word list and populate database
		self._setup_wordlist()
		english_words = self.wordlist.load_words()
		logging.info(f"Word list loaded with {len(english_words):,} words")
		self._import_wordlist()
		
		# Make sure to commit changes
		self.conn.commit()

		c = self.conn.cursor()
		
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
			words = [w for w in all_words if self.wordlist.is_valid_word(w)]
			
			if len(words) < 2:
				continue
					# Count starters
			if words[0]:
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
			
			is_starter = starter_counts.get(word, 0) / count > 0.1
			first_letter = word[0]
			first_two = word[:2] if len(word) >= 2 else None
			first_three = word[:3] if len(word) >= 3 else None
			
			word_data.append((word, count, is_starter, first_letter, first_two, first_three))
		
		c.executemany("INSERT OR REPLACE INTO words VALUES (?, ?, ?, ?, ?, ?)", word_data)
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
		
		self.conn.commit()
		# Don't close self.conn as it will be used later

	def _collect_sentences(self, target_sentences: int = 10000) -> List[str]:
		"""
		Collect sentences for training the language model.
		
		Args:
			target_sentences: Number of sentences to collect
			
		Returns:
			List of sentences for training
		"""
		logging.info(f"Collecting {target_sentences} sentences for training...")
		
		try:
			from datasets import load_dataset
			
			sentences = []
			dataset = load_dataset("daily_dialog", split="train", streaming=True)
			
			# Take only what we need
			for dialog in dataset.take(target_sentences // 5):  # Estimate ~5 sentences per dialog
				for utterance in dialog["dialog"]:
					for sentence in re.split(r'(?<=[.!?])\s+', utterance):
						if sentence and len(sentence.split()) >= 2:
							sentences.append(sentence)
							if len(sentences) >= target_sentences:
								return sentences
			
			return sentences
		except Exception as e:
			logging.error(f"Error collecting sentences: {e}")
			# Fallback to a small set of example sentences
			return [
				"The quick brown fox jumps over the lazy dog",
				"Hello, how are you doing today?",
				"I would like to talk about this project",
				"Python is a great programming language",
				"Machine learning models can predict text"
			]
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
		"""		# Check if tables exist first
		c = self.conn.cursor()
		c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ngrams'")
		if not c.fetchone():
			# Tables don't exist, return empty results
			return [], {'error': 'Database tables not initialized'}
			
		suggestions = Counter()
		debug_info = {'trigram_hits': 0, 'bigram_hits': 0, 'unigram_hits': 0}
		
		# Convert to hashable types
		context_tuple = tuple(w.lower() for w in context_words)
		partial_lower = partial_word.lower()
		
		# 1. Try cached trigram search
		if len(context_tuple) >= 2:
			trigram_context = f"{context_tuple[-2]} {context_tuple[-1]}"
			trigram_results = self._cached_ngram_query(
				'trigram', 
				trigram_context, 
				partial_lower,
				max_suggestions
			)
			for word, freq in trigram_results:
				suggestions[word] += freq
				debug_info['trigram_hits'] += 1

		# 2. Cached bigram search
		if len(suggestions) < max_suggestions and len(context_tuple) >= 1:
			bigram_context = context_tuple[-1]
			bigram_results = self._cached_ngram_query(
				'bigram',
				bigram_context,
				partial_lower,
				max_suggestions - len(suggestions)
			)
			for word, freq in bigram_results:
				if word not in suggestions:
					suggestions[word] += freq
					debug_info['bigram_hits'] += 1

		# 3. Cached unigram (frequent words) search as fallback
		if len(suggestions) < max_suggestions:
			unigram_results = self._cached_word_query(
				partial_lower,
				max_suggestions - len(suggestions)
			)
			for word, freq in unigram_results:
				if word not in suggestions:
					suggestions[word] += freq
					debug_info['unigram_hits'] += 1
		
		sorted_suggestions = [word for word, _ in suggestions.most_common(max_suggestions)]
		
		return sorted_suggestions, debug_info
	
	def _query_ngrams(self, ngram_type: str, context: str, partial: str, limit: int) -> tuple:
		"""Cached n-gram query (returns tuple for hashability)."""
		c = self.conn.cursor()
		query = "SELECT word, frequency FROM ngrams WHERE type = ? AND context = ?"
		params = [ngram_type, context]
		
		if partial:
			query += " AND word LIKE ?"
			params.append(f"{partial}%")
		
		query += " ORDER BY frequency DESC LIMIT ?"
		params.append(limit)
		
		results = c.execute(query, params).fetchall()
		return tuple((row['word'], row['frequency']) for row in results)
	
	def _query_words(self, partial: str, limit: int) -> tuple:
		"""Cached word frequency query (returns tuple for hashability)."""
		c = self.conn.cursor()
		query = "SELECT word, frequency FROM words WHERE 1=1"
		params = []
		
		if partial:
			query += " AND word LIKE ?"
			params.append(f"{partial}%")
		
		query += " ORDER BY frequency DESC LIMIT ?"
		params.append(limit)
		
		results = c.execute(query, params).fetchall()
		return tuple((row['word'], row['frequency']) for row in results)