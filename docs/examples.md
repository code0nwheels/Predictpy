# Examples

Real-world examples and usage patterns for Predictpy.

## Basic Examples

### Simple Text Completion

```python
from predictpy import Predictpy

# Initialize predictor
predictor = Predictpy()

# Basic word prediction
text = "I want to"
suggestions = predictor.predict(text)
print(f"After '{text}': {suggestions}")
# Output: After 'I want to': ['go', 'be', 'see', 'make', 'do']

# Partial word completion
text = "I want to g"
suggestions = predictor.predict(text)
print(f"After '{text}': {suggestions}")
# Output: After 'I want to g': ['go', 'get', 'give', 'good', 'going']
```

### Learning from User Input

```python
# Simulate user typing with learning
predictor = Predictpy()

def simulate_typing_session():
    contexts_and_selections = [
        ("Hello", "world"),
        ("Good", "morning"),
        ("How are", "you"),
        ("Thank you", "very"),
        ("I am", "fine")
    ]
    
    for context, word in contexts_and_selections:
        # Get predictions
        suggestions = predictor.predict(context)
        print(f"Context: '{context}' -> Suggestions: {suggestions}")
        
        # Record user selection
        predictor.select(context, word)
        print(f"User selected: '{word}'")
        print()

simulate_typing_session()
```

---

## Email Assistant

### Smart Email Completion

```python
class EmailAssistant:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True)
        self.train_on_email_patterns()
    
    def train_on_email_patterns(self):
        """Train on common email patterns."""
        email_samples = [
            "Thank you for your email. I will get back to you shortly.",
            "I hope this email finds you well. I wanted to follow up on our previous conversation.",
            "Please let me know if you have any questions or concerns.",
            "I look forward to hearing from you soon.",
            "Thank you for your time and consideration.",
            "I apologize for the delay in my response.",
            "Could you please provide more information about this matter?",
            "I would be happy to schedule a meeting to discuss this further."
        ]
        
        for sample in email_samples:
            self.predictor.learn_from_text(sample, text_type="email", tags=["professional"])
    
    def suggest_email_completion(self, partial_text, email_type="business"):
        """Suggest email completions based on context."""
        context = {
            "text_type": "email",
            "formality": "business" if email_type == "business" else "casual",
            "purpose": self.detect_email_purpose(partial_text)
        }
        
        # Try semantic completion first
        completions = self.predictor.predict_completion(
            partial_text,
            context=context,
            min_words=5
        )
        
        if completions:
            return [comp['text'] for comp in completions[:3]]
        
        # Fallback to word prediction
        return self.predictor.predict(partial_text.split()[-3:], count=5)
    
    def detect_email_purpose(self, text):
        """Simple email purpose detection."""
        text_lower = text.lower()
        if "thank" in text_lower:
            return "gratitude"
        elif "sorry" in text_lower or "apologize" in text_lower:
            return "apology"
        elif "schedule" in text_lower or "meeting" in text_lower:
            return "scheduling"
        elif "follow up" in text_lower:
            return "follow_up"
        else:
            return "general"

# Usage example
assistant = EmailAssistant()

# Test different email contexts
test_cases = [
    "Thank you for your",
    "I apologize for the",
    "Could you please",
    "I look forward to"
]

for test in test_cases:
    suggestions = assistant.suggest_email_completion(test)
    print(f"'{test}' -> {suggestions}")
    print()
```

---

## Chat Application

### Smart Reply Suggestions

```python
class ChatPredictor:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True)
        self.conversation_history = []
        self.train_on_chat_patterns()
    
    def train_on_chat_patterns(self):
        """Train on casual conversation patterns."""
        chat_samples = [
            "Hey! How's it going?",
            "Thanks for reaching out!",
            "I'll get back to you soon.",
            "Sounds good to me!",
            "Let me know what you think.",
            "That's awesome news!",
            "I'm sorry to hear that.",
            "Hope you're having a great day!",
            "Talk to you later!",
            "No problem at all!"
        ]
        
        for sample in chat_samples:
            self.predictor.learn_from_text(sample, text_type="chat", tags=["casual", "friendly"])
    
    def suggest_replies(self, incoming_message, user_style="friendly"):
        """Suggest replies to incoming messages."""
        # Analyze incoming message
        context = self.analyze_message_context(incoming_message)
        context.update({
            "text_type": "chat",
            "formality": "casual",
            "style": user_style
        })
        
        # Generate reply starters based on message type
        if context["type"] == "question":
            starters = ["Yes,", "I think", "Actually,"]
        elif context["type"] == "greeting":
            starters = ["Hey!", "Hi there!", "Hello!"]
        elif context["type"] == "gratitude":
            starters = ["You're welcome!", "No problem!", "Happy to help!"]
        else:
            starters = ["I see.", "That's interesting.", "Thanks for sharing."]
        
        suggestions = []
        for starter in starters:
            completions = self.predictor.predict_completion(
                starter,
                context=context,
                min_words=3
            )
            if completions:
                suggestions.append(starter + " " + completions[0]['text'])
            else:
                # Fallback to word prediction
                words = self.predictor.predict(starter.split(), count=3)
                if words:
                    suggestions.append(starter + " " + words[0])
        
        return suggestions[:3]
    
    def analyze_message_context(self, message):
        """Analyze incoming message for context."""
        message_lower = message.lower()
        
        if "?" in message:
            msg_type = "question"
        elif any(greeting in message_lower for greeting in ["hi", "hey", "hello"]):
            msg_type = "greeting"
        elif any(thanks in message_lower for thanks in ["thank", "thanks"]):
            msg_type = "gratitude"
        elif any(neg in message_lower for neg in ["sorry", "sad", "bad"]):
            msg_type = "negative"
        else:
            msg_type = "statement"
        
        return {
            "type": msg_type,
            "length": len(message.split()),
            "sentiment": self.detect_sentiment(message)
        }
    
    def detect_sentiment(self, message):
        """Simple sentiment detection."""
        positive_words = ["good", "great", "awesome", "happy", "love", "excellent"]
        negative_words = ["bad", "sad", "terrible", "hate", "awful", "sorry"]
        
        words = message.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

# Usage example
chat_bot = ChatPredictor()

# Test different incoming messages
test_messages = [
    "Hey! How are you doing today?",
    "Thanks for your help with the project!",
    "I'm having trouble with this problem.",
    "That movie was really awesome!",
    "Can you help me with something?"
]

for message in test_messages:
    replies = chat_bot.suggest_replies(message)
    print(f"Incoming: '{message}'")
    print("Suggested replies:")
    for i, reply in enumerate(replies, 1):
        print(f"  {i}. {reply}")
    print()
```

---

## Text Editor Integration

### Auto-Completion System

```python
class TextEditorPredictor:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True, training_size="large")
        self.document_type = "general"
        self.user_vocabulary = set()
    
    def set_document_type(self, doc_type):
        """Set document type for better predictions."""
        self.document_type = doc_type
        
        # Load type-specific training data
        if doc_type == "code":
            self.train_on_code_patterns()
        elif doc_type == "academic":
            self.train_on_academic_patterns()
        elif doc_type == "creative":
            self.train_on_creative_patterns()
    
    def train_on_code_patterns(self):
        """Train on code documentation patterns."""
        code_samples = [
            "This function returns the result of the calculation.",
            "Initialize the variable with the default value.",
            "Check if the condition is true before proceeding.",
            "Iterate through the list of items and process each one.",
            "Handle the exception gracefully and log the error."
        ]
        
        for sample in code_samples:
            self.predictor.learn_from_text(sample, text_type="technical", tags=["documentation"])
    
    def get_suggestions(self, current_text, cursor_position, suggestion_type="auto"):
        """Get suggestions based on cursor position and context."""
        
        # Extract context around cursor
        before_cursor = current_text[:cursor_position]
        after_cursor = current_text[cursor_position:]
        
        # Determine suggestion strategy
        if suggestion_type == "word":
            return self.get_word_suggestions(before_cursor)
        elif suggestion_type == "line":
            return self.get_line_completions(before_cursor)
        elif suggestion_type == "paragraph":
            return self.get_paragraph_completions(before_cursor)
        else:
            # Auto-detect best suggestion type
            return self.auto_suggest(before_cursor, after_cursor)
    
    def auto_suggest(self, before_cursor, after_cursor):
        """Automatically determine best suggestion type."""
        words_before = before_cursor.split()
        
        # If in the middle of a word, suggest word completions
        if before_cursor and not before_cursor.endswith(' '):
            partial_word = words_before[-1] if words_before else ""
            context = words_before[:-1] if len(words_before) > 1 else []
            return self.predictor.predict(context + [partial_word], count=5)
        
        # If at end of sentence, suggest sentence completions
        elif before_cursor.endswith('.') or before_cursor.endswith('!') or before_cursor.endswith('?'):
            return self.get_paragraph_completions(before_cursor)
        
        # Default to word prediction
        else:
            return self.get_word_suggestions(before_cursor)
    
    def get_word_suggestions(self, context_text):
        """Get next word suggestions."""
        words = context_text.strip().split()
        context = words[-3:] if len(words) >= 3 else words
        return self.predictor.predict(context, count=5)
    
    def get_line_completions(self, context_text):
        """Get line completion suggestions."""
        # Get last sentence for context
        sentences = context_text.split('.')
        last_sentence = sentences[-1].strip() if sentences else ""
        
        if self.predictor.has_semantic:
            completions = self.predictor.predict_completion(
                last_sentence,
                context={"text_type": self.document_type},
                expected_length="sentence",
                min_words=3
            )
            return [comp['text'] for comp in completions[:3]]
        
        # Fallback to word prediction
        return self.get_word_suggestions(context_text)
    
    def get_paragraph_completions(self, context_text):
        """Get paragraph completion suggestions."""
        if not self.predictor.has_semantic:
            return []
        
        # Use last paragraph as context
        paragraphs = context_text.split('\n\n')
        last_paragraph = paragraphs[-1].strip() if paragraphs else context_text
        
        completions = self.predictor.predict_completion(
            last_paragraph,
            context={"text_type": self.document_type},
            expected_length="paragraph",
            min_words=10
        )
        
        return [comp['text'] for comp in completions[:2]]
    
    def learn_from_document(self, document_text):
        """Learn from the current document."""
        self.predictor.learn_from_text(
            document_text,
            text_type=self.document_type,
            tags=["user_document"]
        )
        
        # Extract vocabulary
        words = document_text.lower().split()
        self.user_vocabulary.update(words)

# Usage example
editor = TextEditorPredictor()

# Simulate document editing
document = """
Artificial intelligence has revolutionized many aspects of modern life.
From healthcare to transportation, AI systems are becoming increasingly
"""

# Set document type
editor.set_document_type("academic")

# Learn from existing content
editor.learn_from_document(document)

# Test suggestions at different positions
cursor_positions = [
    len(document),  # End of document
    document.find("From") + 4,  # Middle of sentence
    document.find("AI") + 2  # After "AI"
]

for pos in cursor_positions:
    suggestions = editor.get_suggestions(document, pos)
    context = document[max(0, pos-20):pos]
    print(f"Context: '...{context}'")
    print(f"Suggestions: {suggestions}")
    print()
```

---

## Writing Assistant

### Creative Writing Helper

```python
class WritingAssistant:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True, training_size="large")
        self.genre = "general"
        self.writing_style = "neutral"
        self.character_voices = {}
    
    def set_genre(self, genre):
        """Set writing genre for better suggestions."""
        self.genre = genre
        self.train_genre_specific()
    
    def train_genre_specific(self):
        """Train on genre-specific patterns."""
        genre_samples = {
            "mystery": [
                "The detective examined the evidence carefully.",
                "Something wasn't right about this case.",
                "The clues led to an unexpected conclusion."
            ],
            "romance": [
                "Her heart skipped a beat when she saw him.",
                "The moment felt perfect and timeless.",
                "Love had a way of surprising people."
            ],
            "sci-fi": [
                "The technology was beyond anything they had seen.",
                "The future held both promise and danger.",
                "Space exploration revealed new mysteries."
            ]
        }
        
        if self.genre in genre_samples:
            for sample in genre_samples[self.genre]:
                self.predictor.learn_from_text(
                    sample,
                    text_type="creative",
                    tags=[self.genre, "fiction"]
                )
    
    def suggest_next_sentence(self, current_text, tone="neutral"):
        """Suggest next sentence for creative writing."""
        context = {
            "text_type": "creative",
            "genre": self.genre,
            "tone": tone,
            "style": self.writing_style
        }
        
        completions = self.predictor.predict_completion(
            current_text,
            context=context,
            expected_length="sentence",
            min_words=8
        )
        
        return [comp['text'] for comp in completions[:3]]
    
    def suggest_paragraph_continuation(self, current_text):
        """Suggest paragraph continuation."""
        context = {
            "text_type": "creative",
            "genre": self.genre,
            "style": self.writing_style
        }
        
        completions = self.predictor.predict_completion(
            current_text,
            context=context,
            expected_length="paragraph",
            min_words=20
        )
        
        return [comp['text'] for comp in completions[:2]]
    
    def suggest_dialogue(self, character_name, situation_context):
        """Suggest dialogue for a character."""
        if character_name in self.character_voices:
            voice_style = self.character_voices[character_name]
        else:
            voice_style = "neutral"
        
        dialogue_starter = f'"{situation_context}'
        
        context = {
            "text_type": "dialogue",
            "genre": self.genre,
            "character_style": voice_style
        }
        
        completions = self.predictor.predict_completion(
            dialogue_starter,
            context=context,
            min_words=5
        )
        
        # Format as dialogue
        suggestions = []
        for comp in completions[:3]:
            dialogue = dialogue_starter + comp['text']
            if not dialogue.endswith('"'):
                dialogue += '"'
            suggestions.append(dialogue)
        
        return suggestions
    
    def add_character_voice(self, name, sample_dialogue):
        """Learn a character's voice from sample dialogue."""
        self.character_voices[name] = "custom"
        
        # Extract dialogue content (remove quotes)
        clean_dialogue = sample_dialogue.replace('"', '').replace("'", "")
        
        self.predictor.learn_from_text(
            clean_dialogue,
            text_type="dialogue",
            tags=[name, "character_voice", self.genre]
        )

# Usage example
writer = WritingAssistant()
writer.set_genre("mystery")

# Add character voice
writer.add_character_voice("Detective Smith", """
"I've seen a lot of cases in my time, but this one's different.
The evidence doesn't add up the way it should.
Something's not right here, and I intend to find out what."
""")

# Test writing suggestions
story_beginning = """
The old mansion stood silently on the hill, its windows dark against the stormy sky.
Detective Smith approached the front door, rain dripping from his coat.
"""

# Get next sentence suggestions
next_sentences = writer.suggest_next_sentence(story_beginning, tone="suspenseful")
print("Next sentence suggestions:")
for i, sentence in enumerate(next_sentences, 1):
    print(f"{i}. {sentence}")
print()

# Get dialogue suggestions
dialogue_suggestions = writer.suggest_dialogue("Detective Smith", "I think we need to")
print("Detective Smith dialogue suggestions:")
for i, dialogue in enumerate(dialogue_suggestions, 1):
    print(f"{i}. {dialogue}")
print()

# Get paragraph continuation
paragraph_suggestions = writer.suggest_paragraph_continuation(story_beginning)
print("Paragraph continuation suggestions:")
for i, paragraph in enumerate(paragraph_suggestions, 1):
    print(f"{i}. {paragraph}")
```

---

## Data Entry Assistant

### Form Completion Helper

```python
class FormAssistant:
    def __init__(self):
        self.predictor = Predictpy()
        self.field_patterns = {}
        self.user_preferences = {}
        self.train_on_common_patterns()
    
    def train_on_common_patterns(self):
        """Train on common form field patterns."""
        patterns = {
            "address": [
                "123 Main Street, New York, NY 10001",
                "456 Oak Avenue, Los Angeles, CA 90210",
                "789 Pine Road, Chicago, IL 60601"
            ],
            "phone": [
                "(555) 123-4567",
                "555-987-6543", 
                "+1 (555) 246-8135"
            ],
            "email": [
                "john.doe@email.com",
                "jane.smith@company.org",
                "user123@domain.net"
            ]
        }
        
        for field_type, examples in patterns.items():
            for example in examples:
                self.predictor.learn_from_text(
                    example,
                    text_type="form_data",
                    tags=[field_type, "personal_info"]
                )
    
    def suggest_field_completion(self, field_type, partial_input, user_context=None):
        """Suggest completion for form fields."""
        
        # Use field-specific patterns
        if field_type in self.field_patterns:
            # Get predictions based on learned patterns
            context = self.field_patterns[field_type][-3:]  # Last 3 examples
            suggestions = self.predictor.predict(context + [partial_input])
        else:
            # General prediction
            suggestions = self.predictor.predict(partial_input.split())
        
        # Apply field-specific formatting
        formatted_suggestions = []
        for suggestion in suggestions:
            formatted = self.format_for_field_type(suggestion, field_type)
            if formatted:
                formatted_suggestions.append(formatted)
        
        return formatted_suggestions[:3]
    
    def format_for_field_type(self, suggestion, field_type):
        """Format suggestion based on field type."""
        if field_type == "phone":
            # Basic phone number formatting
            digits = ''.join(filter(str.isdigit, suggestion))
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        elif field_type == "email":
            # Basic email validation
            if "@" in suggestion and "." in suggestion:
                return suggestion.lower()
        
        elif field_type == "address":
            # Title case for addresses
            return suggestion.title()
        
        elif field_type == "name":
            # Title case for names
            return suggestion.title()
        
        return suggestion
    
    def learn_user_preference(self, field_type, value):
        """Learn user's preferences for specific fields."""
        if field_type not in self.user_preferences:
            self.user_preferences[field_type] = []
        
        self.user_preferences[field_type].append(value)
        
        # Also train the predictor
        self.predictor.learn_from_text(
            value,
            text_type="user_preference",
            tags=[field_type, "personal"]
        )
    
    def get_frequent_values(self, field_type):
        """Get frequently used values for a field type."""
        if field_type in self.user_preferences:
            from collections import Counter
            counter = Counter(self.user_preferences[field_type])
            return [value for value, count in counter.most_common(3)]
        return []

# Usage example
form_helper = FormAssistant()

# Simulate form filling with learning
form_sessions = [
    ("name", "John", "John Doe"),
    ("email", "john", "john.doe@email.com"),
    ("phone", "555", "555-123-4567"),
    ("address", "123", "123 Main Street, New York, NY 10001")
]

print("Form completion suggestions:")
for field_type, partial, full_value in form_sessions:
    # Get suggestions
    suggestions = form_helper.suggest_field_completion(field_type, partial)
    print(f"\nField: {field_type}")
    print(f"Partial input: '{partial}'")
    print(f"Suggestions: {suggestions}")
    
    # Learn from user's actual input
    form_helper.learn_user_preference(field_type, full_value)
    print(f"User selected: '{full_value}'")

# Test frequent values
print(f"\nFrequent email values: {form_helper.get_frequent_values('email')}")
print(f"Frequent phone values: {form_helper.get_frequent_values('phone')}")
```

---

## Performance Testing

### Benchmark Different Configurations

```python
import time
from predictpy import Predictpy

def benchmark_configuration(config_name, config, test_phrases):
    """Benchmark a specific configuration."""
    print(f"\nTesting {config_name}:")
    print(f"Config: {config}")
    
    # Initialize with timing
    start_time = time.time()
    predictor = Predictpy(config=config)
    init_time = time.time() - start_time
    
    # Test predictions
    prediction_times = []
    for phrase in test_phrases:
        start_time = time.time()
        suggestions = predictor.predict(phrase)
        prediction_times.append(time.time() - start_time)
    
    # Test semantic completion if available
    completion_times = []
    if predictor.has_semantic:
        for phrase in test_phrases:
            start_time = time.time()
            completions = predictor.predict_completion(phrase)
            completion_times.append(time.time() - start_time)
    
    # Report results
    print(f"  Initialization time: {init_time:.3f}s")
    print(f"  Average prediction time: {sum(prediction_times)/len(prediction_times):.3f}s")
    if completion_times:
        print(f"  Average completion time: {sum(completion_times)/len(completion_times):.3f}s")
    print(f"  Semantic features: {'Available' if predictor.has_semantic else 'Not available'}")
    
    return {
        'config_name': config_name,
        'init_time': init_time,
        'avg_prediction_time': sum(prediction_times)/len(prediction_times),
        'avg_completion_time': sum(completion_times)/len(completion_times) if completion_times else None,
        'has_semantic': predictor.has_semantic
    }

# Test configurations
configs = {
    "Lightweight": {
        "training_size": "small",
        "use_semantic": False,
        "auto_train": True
    },
    "Balanced": {
        "training_size": "medium", 
        "use_semantic": True,
        "auto_train": True
    },
    "High-Performance": {
        "training_size": "large",
        "use_semantic": True,
        "auto_train": True
    }
}

test_phrases = [
    "Hello world",
    "I want to",
    "Thank you for",
    "How are you",
    "Good morning"
]

# Run benchmarks
results = []
for config_name, config in configs.items():
    try:
        result = benchmark_configuration(config_name, config, test_phrases)
        results.append(result)
    except Exception as e:
        print(f"Failed to test {config_name}: {e}")

# Summary
print("\n" + "="*50)
print("BENCHMARK SUMMARY")
print("="*50)
for result in results:
    print(f"{result['config_name']:15} | Init: {result['init_time']:.3f}s | Predict: {result['avg_prediction_time']:.3f}s | Semantic: {'Yes' if result['has_semantic'] else 'No'}")
```

---

## Performance Examples

### Optimized Caching

This example demonstrates how to use the caching system for optimal performance:

```python
"""
Example usage of Predictpy with optimized caching.
"""
import time
from predictpy import Predictpy, calculate_optimal_cache_size

def main():
    # Initialize with optimal cache sizes
    cache_sizes = calculate_optimal_cache_size()
    print(f"Using cache sizes: {cache_sizes}")
    
    predictor = Predictpy(
        config={'cache_config': cache_sizes}
    )
    
    # Monitor cache performance
    print("Running prediction tests...")
    for i in range(100):
        # Mix of cached and new queries
        if i % 3 == 0:
            predictor.predict("I want to")  # Should be cached
        else:
            predictor.predict(f"Test phrase {i}")  # New query
    
    # Check cache performance
    cache_info = predictor.cache_info
    print(f"Cache hit rate: {cache_info['predict_cache']['hit_rate']:.2%}")
    print(f"Cache size: {cache_info['predict_cache']['currsize']}/{cache_info['predict_cache']['maxsize']}")
    
    # Test invalidation
    print("\nTesting cache invalidation...")
    for i in range(60):
        predictor.select("I like", "to")
    
    # Check modification counters
    print(f"Modifications since clear: {predictor.cache_info['modifications_since_clear']}")
    
    # Force clear all caches
    print("\nForcing cache clear...")
    predictor.clear_all_caches()
    print(f"Cache after clear - size: {predictor.cache_info['predict_cache']['currsize']}")

if __name__ == "__main__":
    main()
```

Expected output:

```
Using cache sizes: {'predict_size': 4096, 'completion_size': 256}
Running prediction tests...
Cache hit rate: 33.00%
Cache size: 68/4096

Testing cache invalidation...
Modifications since clear: 10

Forcing cache clear...
Cache after clear - size: 0
```

This completes the examples documentation with practical, real-world usage patterns for different scenarios. Each example is self-contained and demonstrates key features of Predictpy in context.
