# Advanced Usage

Advanced features and techniques for power users of Predictpy.

## Custom Training Data

### Training with Domain-Specific Corpora

```python
from predictpy import Predictpy
import os

class DomainSpecificPredictor:
    def __init__(self, domain="general"):
        self.domain = domain
        self.predictor = Predictpy(
            training_size="large",
            use_semantic=True,
            auto_train=False  # We'll do custom training
        )
        self.train_domain_specific()
    
    def train_domain_specific(self):
        """Train on domain-specific data."""
        domain_texts = {
            "medical": [
                "The patient presents with symptoms of acute inflammation.",
                "Diagnosis indicates chronic conditions requiring treatment.",
                "Medication dosage should be adjusted based on patient response.",
                "Follow-up appointments are scheduled for monitoring progress."
            ],
            "legal": [
                "The contract stipulates terms and conditions for both parties.",
                "Legal precedent establishes the framework for this case.",
                "Evidence presented supports the plaintiff's claims.",
                "The court hereby orders the following remedies."
            ],
            "technical": [
                "The system architecture implements scalable microservices design.",
                "Performance optimization requires careful resource allocation.",
                "Database indexing improves query execution speed significantly.",
                "Error handling ensures robust application behavior."
            ]
        }
        
        if self.domain in domain_texts:
            for text in domain_texts[self.domain]:
                self.predictor.learn_from_text(
                    text,
                    text_type=self.domain,
                    tags=["domain_specific", "training"]
                )
    
    def load_corpus_from_file(self, file_path, chunk_size=1000):
        """Load and train from large text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    combined_text = " ".join(chunk)
                    self.predictor.learn_from_text(
                        combined_text,
                        text_type=self.domain,
                        tags=["corpus", "file_training"]
                    )
                    chunk = []
            
            # Process remaining chunk
            if chunk:
                combined_text = " ".join(chunk)
                self.predictor.learn_from_text(
                    combined_text,
                    text_type=self.domain,
                    tags=["corpus", "file_training"]
                )

# Usage
medical_predictor = DomainSpecificPredictor("medical")
# medical_predictor.load_corpus_from_file("medical_texts.txt")

# Test domain-specific predictions
medical_suggestions = medical_predictor.predictor.predict("The patient")
print(f"Medical domain suggestions: {medical_suggestions}")
```

### Multi-Language Support

```python
class MultiLanguagePredictor:
    def __init__(self):
        self.predictors = {}
        self.current_language = "en"
        
    def add_language(self, lang_code, training_data=None):
        """Add support for a new language."""
        # Create separate database for each language
        db_path = f"~/.predictpy/predictpy_{lang_code}.db"
        
        self.predictors[lang_code] = Predictpy(
            db_path=db_path,
            use_semantic=True,
            training_size="medium"
        )
        
        if training_data:
            for text in training_data:
                self.predictors[lang_code].learn_from_text(
                    text,
                    text_type="multilingual",
                    tags=[lang_code, "training"]
                )
    
    def set_language(self, lang_code):
        """Switch to a specific language."""
        if lang_code in self.predictors:
            self.current_language = lang_code
        else:
            raise ValueError(f"Language {lang_code} not supported")
    
    def predict(self, text, count=5):
        """Predict in current language."""
        return self.predictors[self.current_language].predict(text, count)
    
    def detect_language(self, text):
        """Simple language detection based on character patterns."""
        # This is a simplified example - use proper language detection libraries
        if any(ord(char) > 127 for char in text):
            # Non-ASCII characters, might be non-English
            if any(char in "àáâãäåçèéêë" for char in text.lower()):
                return "fr"  # French
            elif any(char in "äöüß" for char in text.lower()):
                return "de"  # German
            elif any(char in "ñáéíóú" for char in text.lower()):
                return "es"  # Spanish
        
        return "en"  # Default to English

# Usage
ml_predictor = MultiLanguagePredictor()

# Add languages with sample training data
ml_predictor.add_language("en", [
    "Hello, how are you today?",
    "Thank you for your help.",
    "I hope you have a great day."
])

ml_predictor.add_language("es", [
    "Hola, ¿cómo estás hoy?",
    "Gracias por tu ayuda.",
    "Espero que tengas un gran día."
])

# Test predictions in different languages
ml_predictor.set_language("en")
english_suggestions = ml_predictor.predict("Hello")
print(f"English: {english_suggestions}")

ml_predictor.set_language("es")
spanish_suggestions = ml_predictor.predict("Hola")
print(f"Spanish: {spanish_suggestions}")
```

---

## Custom Neural Models

### Integrating Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class CustomSemanticPredictor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.predictor = Predictpy(use_semantic=False)  # Disable built-in semantic
        self.embedding_model = SentenceTransformer(model_name)
        self.semantic_memory = []
        self.embeddings = []
    
    def learn_with_custom_embeddings(self, text, metadata=None):
        """Learn text with custom embedding processing."""
        # Generate embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Store text and embedding
        self.semantic_memory.append({
            'text': text,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        self.embeddings.append(embedding)
        
        # Also train word-level predictor
        self.predictor.learn_from_text(text)
    
    def find_similar_contexts(self, query, top_k=5):
        """Find similar contexts using cosine similarity."""
        if not self.embeddings:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for i, stored_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, similarity in similarities[:top_k]:
            results.append({
                'text': self.semantic_memory[i]['text'],
                'similarity': similarity,
                'metadata': self.semantic_memory[i]['metadata']
            })
        
        return results
    
    def predict_with_context(self, text, use_semantic=True):
        """Predict using both word-level and semantic context."""
        # Get word-level predictions
        word_predictions = self.predictor.predict(text)
        
        if not use_semantic or not self.embeddings:
            return word_predictions
        
        # Get semantic context
        similar_contexts = self.find_similar_contexts(text, top_k=3)
        
        # Combine predictions (simple approach)
        semantic_suggestions = []
        for context in similar_contexts:
            context_words = context['text'].split()
            # Find words that might follow the query
            for word in context_words:
                if word.lower() not in text.lower() and len(word) > 2:
                    semantic_suggestions.append(word)
        
        # Merge and deduplicate
        all_suggestions = word_predictions + semantic_suggestions[:3]
        unique_suggestions = []
        seen = set()
        for suggestion in all_suggestions:
            if suggestion.lower() not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion.lower())
        
        return unique_suggestions[:5]

# Usage
import time
custom_predictor = CustomSemanticPredictor()

# Train with custom metadata
training_samples = [
    ("Hello world, how are you today?", {"type": "greeting", "formality": "casual"}),
    ("Good morning, I hope you slept well.", {"type": "greeting", "formality": "polite"}),
    ("Thank you for your assistance.", {"type": "gratitude", "formality": "formal"}),
    ("Thanks a lot for helping me out!", {"type": "gratitude", "formality": "casual"})
]

for text, metadata in training_samples:
    custom_predictor.learn_with_custom_embeddings(text, metadata)

# Test predictions
test_query = "Good morning"
similar = custom_predictor.find_similar_contexts(test_query)
print("Similar contexts:")
for context in similar:
    print(f"  '{context['text']}' (similarity: {context['similarity']:.3f})")

predictions = custom_predictor.predict_with_context(test_query)
print(f"Predictions for '{test_query}': {predictions}")
```

---

## Performance Optimization

### Caching and Memory Management

```python
from functools import lru_cache
import pickle
import os

class OptimizedPredictor:
    def __init__(self, cache_size=1000):
        self.predictor = Predictpy(use_semantic=True)
        self.cache_size = cache_size
        self.prediction_cache = {}
        self.cache_file = "prediction_cache.pkl"
        self.load_cache()
    
    @lru_cache(maxsize=1000)
    def cached_predict(self, text_tuple, count=5):
        """Cached prediction using LRU cache."""
        text = " ".join(text_tuple) if isinstance(text_tuple, tuple) else text_tuple
        return tuple(self.predictor.predict(text, count))
    
    def predict_with_cache(self, text, count=5):
        """Predict with custom caching logic."""
        cache_key = f"{text}:{count}"
        
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Make prediction
        result = self.predictor.predict(text, count)
        
        # Cache result
        if len(self.prediction_cache) < self.cache_size:
            self.prediction_cache[cache_key] = result
        
        return result
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.prediction_cache, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.prediction_cache = pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}")
            self.prediction_cache = {}
    
    def clear_cache(self):
        """Clear all caches."""
        self.prediction_cache.clear()
        self.cached_predict.cache_clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
    
    def get_cache_stats(self):
        """Get cache performance statistics."""
        return {
            'cache_size': len(self.prediction_cache),
            'lru_cache_info': self.cached_predict.cache_info(),
            'cache_hit_ratio': self.cached_predict.cache_info().hits / 
                             max(1, self.cached_predict.cache_info().hits + 
                                 self.cached_predict.cache_info().misses)
        }

# Usage
optimizer = OptimizedPredictor()

# Test with repeated queries
test_phrases = ["Hello", "How are", "Thank you"] * 3

for phrase in test_phrases:
    # This will benefit from caching on repeated queries
    result = optimizer.predict_with_cache(phrase)
    print(f"'{phrase}' -> {result}")

# Check cache performance
stats = optimizer.get_cache_stats()
print(f"Cache statistics: {stats}")

# Save cache for next session
optimizer.save_cache()
```

### Batch Processing

```python
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

class BatchPredictor:
    def __init__(self, batch_size=10, max_workers=4):
        self.predictor = Predictpy(use_semantic=True)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.request_queue = Queue()
        self.result_callbacks = {}
        self.is_running = False
        
    def start_batch_processor(self):
        """Start the batch processing thread."""
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._batch_processor)
        self.processor_thread.start()
    
    def stop_batch_processor(self):
        """Stop the batch processing thread."""
        self.is_running = False
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
    
    def _batch_processor(self):
        """Background thread for batch processing."""
        batch = []
        
        while self.is_running:
            try:
                # Collect batch
                while len(batch) < self.batch_size and self.is_running:
                    try:
                        item = self.request_queue.get(timeout=0.1)
                        batch.append(item)
                    except:
                        break
                
                if batch:
                    self._process_batch(batch)
                    batch = []
                    
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    def _process_batch(self, batch):
        """Process a batch of requests."""
        def process_single(item):
            request_id, text, count, callback = item
            try:
                result = self.predictor.predict(text, count)
                if callback:
                    callback(request_id, result, None)
            except Exception as e:
                if callback:
                    callback(request_id, None, e)
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(process_single, batch)
    
    def predict_async(self, text, count=5, callback=None):
        """Add prediction request to batch queue."""
        import uuid
        request_id = str(uuid.uuid4())
        
        self.request_queue.put((request_id, text, count, callback))
        return request_id
    
    def predict_batch_sync(self, text_list, count=5):
        """Process multiple predictions synchronously."""
        results = {}
        completed = threading.Event()
        remaining = len(text_list)
        
        def result_callback(request_id, result, error):
            nonlocal remaining
            results[request_id] = (result, error)
            remaining -= 1
            if remaining == 0:
                completed.set()
        
        # Submit all requests
        request_ids = []
        for text in text_list:
            request_id = self.predict_async(text, count, result_callback)
            request_ids.append(request_id)
        
        # Wait for completion
        completed.wait(timeout=30)  # 30 second timeout
        
        # Return results in order
        ordered_results = []
        for request_id in request_ids:
            if request_id in results:
                result, error = results[request_id]
                if error:
                    ordered_results.append([])
                else:
                    ordered_results.append(result)
            else:
                ordered_results.append([])  # Timeout
        
        return ordered_results

# Usage
batch_predictor = BatchPredictor(batch_size=5, max_workers=2)
batch_predictor.start_batch_processor()

# Test batch processing
test_texts = [
    "Hello world",
    "How are you",
    "Thank you for",
    "Good morning",
    "I hope you"
]

# Synchronous batch processing
results = batch_predictor.predict_batch_sync(test_texts)
for text, result in zip(test_texts, results):
    print(f"'{text}' -> {result}")

# Asynchronous processing with callbacks
def print_result(request_id, result, error):
    if error:
        print(f"Error for {request_id}: {error}")
    else:
        print(f"Async result for {request_id}: {result}")

for text in test_texts:
    batch_predictor.predict_async(text, callback=print_result)

# Allow time for async processing
import time
time.sleep(2)

batch_predictor.stop_batch_processor()
```

---

## Integration Patterns

### Plugin Architecture

```python
from abc import ABC, abstractmethod
import importlib

class PredictionPlugin(ABC):
    """Base class for prediction plugins."""
    
    @abstractmethod
    def name(self) -> str:
        """Return plugin name."""
        pass
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text before prediction."""
        pass
    
    @abstractmethod
    def postprocess(self, predictions: list) -> list:
        """Postprocess predictions."""
        pass
    
    @abstractmethod
    def can_handle(self, context: dict) -> bool:
        """Check if plugin can handle this context."""
        pass

class PluginManager:
    def __init__(self):
        self.plugins = []
        self.predictor = Predictpy(use_semantic=True)
    
    def register_plugin(self, plugin: PredictionPlugin):
        """Register a new plugin."""
        self.plugins.append(plugin)
        print(f"Registered plugin: {plugin.name()}")
    
    def predict_with_plugins(self, text: str, context: dict = None, count: int = 5):
        """Make predictions using applicable plugins."""
        context = context or {}
        
        # Find applicable plugins
        applicable_plugins = [p for p in self.plugins if p.can_handle(context)]
        
        processed_text = text
        
        # Apply preprocessing
        for plugin in applicable_plugins:
            processed_text = plugin.preprocess(processed_text)
        
        # Make prediction
        predictions = self.predictor.predict(processed_text, count)
        
        # Apply postprocessing
        for plugin in applicable_plugins:
            predictions = plugin.postprocess(predictions)
        
        return predictions

# Example plugins
class EmailPlugin(PredictionPlugin):
    def name(self) -> str:
        return "EmailPlugin"
    
    def preprocess(self, text: str) -> str:
        # Add email-specific context
        if not text.endswith(' '):
            text += ' '
        return text
    
    def postprocess(self, predictions: list) -> list:
        # Filter for email-appropriate words
        email_words = [p for p in predictions if '@' not in p and 'http' not in p]
        return email_words
    
    def can_handle(self, context: dict) -> bool:
        return context.get('type') == 'email'

class CodePlugin(PredictionPlugin):
    def name(self) -> str:
        return "CodePlugin"
    
    def preprocess(self, text: str) -> str:
        # Handle code context
        return text.lower()
    
    def postprocess(self, predictions: list) -> list:
        # Prioritize programming keywords
        code_keywords = ['function', 'return', 'if', 'else', 'for', 'while', 'class']
        prioritized = [p for p in predictions if p in code_keywords]
        regular = [p for p in predictions if p not in code_keywords]
        return prioritized + regular
    
    def can_handle(self, context: dict) -> bool:
        return context.get('type') == 'code'

# Usage
plugin_manager = PluginManager()
plugin_manager.register_plugin(EmailPlugin())
plugin_manager.register_plugin(CodePlugin())

# Test with different contexts
email_context = {'type': 'email'}
code_context = {'type': 'code'}

email_predictions = plugin_manager.predict_with_plugins(
    "Thank you for", 
    context=email_context
)
print(f"Email predictions: {email_predictions}")

code_predictions = plugin_manager.predict_with_plugins(
    "def process", 
    context=code_context
)
print(f"Code predictions: {code_predictions}")
```

### Event-Driven Architecture

```python
import threading
from typing import Callable, Dict, List

class PredictionEvent:
    def __init__(self, event_type: str, data: dict):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

class EventDrivenPredictor:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True)
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue = Queue()
        self.is_processing = False
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit(self, event_type: str, data: dict):
        """Emit an event."""
        event = PredictionEvent(event_type, data)
        self.event_queue.put(event)
        
        if not self.is_processing:
            self.start_event_processor()
    
    def start_event_processor(self):
        """Start processing events in background."""
        self.is_processing = True
        threading.Thread(target=self._process_events, daemon=True).start()
    
    def _process_events(self):
        """Process events from queue."""
        while self.is_processing:
            try:
                event = self.event_queue.get(timeout=1)
                self._handle_event(event)
            except:
                continue
    
    def _handle_event(self, event: PredictionEvent):
        """Handle a single event."""
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Event handler error: {e}")
    
    def predict(self, text: str, count: int = 5):
        """Make prediction and emit events."""
        # Emit prediction request event
        self.emit('prediction_requested', {
            'text': text,
            'count': count
        })
        
        # Make prediction
        result = self.predictor.predict(text, count)
        
        # Emit prediction result event
        self.emit('prediction_completed', {
            'text': text,
            'predictions': result,
            'count': len(result)
        })
        
        return result
    
    def learn_from_selection(self, context: str, word: str):
        """Learn from selection and emit events."""
        # Emit learning event
        self.emit('learning_started', {
            'context': context,
            'word': word
        })
        
        # Perform learning
        self.predictor.select(context, word)
        
        # Emit completion event
        self.emit('learning_completed', {
            'context': context,
            'word': word
        })

# Example event handlers
def log_predictions(event: PredictionEvent):
    """Log all predictions."""
    if event.event_type == 'prediction_completed':
        data = event.data
        print(f"LOG: Predicted for '{data['text']}': {data['predictions']}")

def analyze_performance(event: PredictionEvent):
    """Analyze prediction performance."""
    if event.event_type == 'prediction_completed':
        data = event.data
        if len(data['predictions']) < 3:
            print(f"WARNING: Low prediction count for '{data['text']}'")

def update_statistics(event: PredictionEvent):
    """Update usage statistics."""
    if event.event_type == 'learning_completed':
        data = event.data
        # Update statistics (simplified)
        print(f"STATS: Learned '{data['word']}' for context '{data['context']}'")

# Usage
event_predictor = EventDrivenPredictor()

# Subscribe to events
event_predictor.subscribe('prediction_completed', log_predictions)
event_predictor.subscribe('prediction_completed', analyze_performance)
event_predictor.subscribe('learning_completed', update_statistics)

# Test predictions with event handling
test_phrases = ["Hello", "Thank you", "How"]

for phrase in test_phrases:
    predictions = event_predictor.predict(phrase)
    # Simulate user selection
    if predictions:
        event_predictor.learn_from_selection(phrase, predictions[0])

# Allow time for event processing
time.sleep(1)
```

---

## Custom Storage Backends

### Redis Backend for Distributed Usage

```python
import redis
import json
import pickle
from typing import Optional

class RedisBackend:
    def __init__(self, host='localhost', port=6379, db=0, prefix='predictpy'):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
    
    def get_key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.prefix}:{key}"
    
    def store_predictions(self, context: str, predictions: list):
        """Store predictions in Redis."""
        key = self.get_key(f"predictions:{hash(context)}")
        self.redis_client.setex(
            key,
            3600,  # 1 hour TTL
            json.dumps(predictions)
        )
    
    def get_predictions(self, context: str) -> Optional[list]:
        """Get cached predictions from Redis."""
        key = self.get_key(f"predictions:{hash(context)}")
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def store_user_selection(self, user_id: str, context: str, word: str):
        """Store user selection in Redis."""
        key = self.get_key(f"selections:{user_id}")
        selection = {
            'context': context,
            'word': word,
            'timestamp': time.time()
        }
        self.redis_client.lpush(key, json.dumps(selection))
        # Keep only last 1000 selections
        self.redis_client.ltrim(key, 0, 999)
    
    def get_user_selections(self, user_id: str, limit: int = 100) -> list:
        """Get user selections from Redis."""
        key = self.get_key(f"selections:{user_id}")
        selections = self.redis_client.lrange(key, 0, limit - 1)
        return [json.loads(s) for s in selections]

class DistributedPredictor:
    def __init__(self, user_id: str, redis_backend: RedisBackend):
        self.user_id = user_id
        self.backend = redis_backend
        self.predictor = Predictpy(use_semantic=True)
        self.load_user_data()
    
    def load_user_data(self):
        """Load user's historical selections."""
        selections = self.backend.get_user_selections(self.user_id)
        for selection in selections:
            self.predictor.select(selection['context'], selection['word'])
    
    def predict(self, text: str, count: int = 5) -> list:
        """Predict with Redis caching."""
        # Check cache first
        cached = self.backend.get_predictions(f"{self.user_id}:{text}")
        if cached:
            return cached[:count]
        
        # Make prediction
        predictions = self.predictor.predict(text, count)
        
        # Cache result
        self.backend.store_predictions(f"{self.user_id}:{text}", predictions)
        
        return predictions
    
    def select(self, context: str, word: str):
        """Record selection with distributed storage."""
        # Store in Redis
        self.backend.store_user_selection(self.user_id, context, word)
        
        # Update local predictor
        self.predictor.select(context, word)

# Usage (requires Redis server)
try:
    redis_backend = RedisBackend()
    
    # Test with multiple users
    user1 = DistributedPredictor("user1", redis_backend)
    user2 = DistributedPredictor("user2", redis_backend)
    
    # User 1 interactions
    predictions1 = user1.predict("Hello")
    user1.select("Hello", predictions1[0] if predictions1 else "world")
    
    # User 2 interactions
    predictions2 = user2.predict("Hello")
    user2.select("Hello", predictions2[0] if predictions2 else "there")
    
    print(f"User 1 predictions: {predictions1}")
    print(f"User 2 predictions: {predictions2}")
    
except Exception as e:
    print(f"Redis backend not available: {e}")
```

This advanced usage guide covers sophisticated techniques for power users who need custom training, performance optimization, plugin architectures, event-driven systems, and distributed storage backends.
