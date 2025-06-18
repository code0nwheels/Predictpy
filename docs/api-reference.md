# API Reference

Complete reference for all Predictpy classes and methods.

## Predictpy Class

The main interface for word prediction and text completion.

### Constructor

```python
Predictpy(config=None, db_path=None, auto_train=True, training_size="medium", use_semantic=True)
```

Initialize the predictor with optional configuration.

**Parameters:**
- `config` *(str | dict, optional)*: Path to config file or config dictionary
- `db_path` *(str, optional)*: Custom database path (defaults to `~/.predictpy/predictpy.db`)
- `auto_train` *(bool)*: Auto-train if database doesn't exist (default: `True`)
- `training_size` *(str)*: Training data size - `"small"` (1k), `"medium"` (10k), or `"large"` (50k sentences). Default: `"medium"`
- `use_semantic` *(bool)*: Enable semantic completion features (default: `True`, requires ChromaDB)

**Example:**
```python
# Basic initialization
predictor = Predictpy()

# Custom configuration
predictor = Predictpy(
    db_path="/custom/path/predictions.db",
    training_size="large",
    use_semantic=True
)

# From config file
predictor = Predictpy(config="/path/to/config.json")
```

---

## Core Methods

### predict()

```python
predict(text, count=5) -> List[str]
```

Get word predictions from text input.

**Parameters:**
- `text` *(str | List[str])*: Input text as string or word list
- `count` *(int)*: Number of predictions to return (default: 5)

**Returns:** List of predicted words

**Examples:**
```python
# String input
predictor.predict("Hello w")  # ['world', 'will', 'was', 'would', 'with']

# List input  
predictor.predict(["Hello"], count=3)  # ['world', 'there', 'everyone']

# Partial word completion
predictor.predict("I want to g")  # ['go', 'get', 'give', 'good', 'going']

# Context-aware prediction
predictor.predict("Thank you for your ")  # ['help', 'email', 'time', 'support']
```

---

### select()

```python
select(context, word, index=None)
```

Record a word selection to improve future predictions.

**Parameters:**
- `context` *(str | List[str])*: Context when word was selected
- `word` *(str)*: The selected word
- `index` *(int, optional)*: Index if selecting from predictions

**Examples:**
```python
# String context
predictor.select("I want", "to")

# List context
predictor.select(["I", "want"], "to", index=0)

# Learning from user choice
suggestions = predictor.predict("Good")  # ['morning', 'day', 'job']
# User selects "morning"
predictor.select("Good", "morning", index=0)
```

---

### learn_from_text()

```python
learn_from_text(text, text_type="general", tags=None)
```

Learn from a block of text to improve future predictions.

**Parameters:**
- `text` *(str)*: Text to learn from
- `text_type` *(str)*: Type of text (email, chat, document, etc.). Default: `"general"`
- `tags` *(List[str], optional)*: Optional tags for categorization

**Examples:**
```python
# Learn from email
predictor.learn_from_text(
    "Thank you for the meeting. It was very productive.",
    text_type="email",
    tags=["business", "meeting"]
)

# Learn from chat messages
predictor.learn_from_text(
    "Hey! How are you doing today? Hope everything is going well.",
    text_type="chat",
    tags=["casual", "greeting"]
)

# Learn from documents
with open('document.txt', 'r') as f:
    content = f.read()
    predictor.learn_from_text(content, text_type="document")
```

---

## Semantic Methods

### predict_completion()

```python
predict_completion(text, min_words=5, context=None, style=None, expected_length=None) -> List[Dict[str, Any]]
```

Get semantic completions for thoughts/paragraphs using AI.

**Parameters:**
- `text` *(str)*: Partial text to complete
- `min_words` *(int)*: Minimum words in completion (default: 5)
- `context` *(dict, optional)*: Context filters (type, sentiment, formality, etc.)
- `style` *(str, optional)*: Style preference (formal, casual, etc.)
- `expected_length` *(str, optional)*: Expected completion length (sentence, paragraph)

**Returns:** List of completion dictionaries with 'text', 'confidence', 'type', etc.

**Examples:**
```python
# Basic completion
completions = predictor.predict_completion("I wanted to let you know that")
for completion in completions:
    print(f"→ {completion['text']} (confidence: {completion['confidence']:.2f})")

# Context-aware completion
completions = predictor.predict_completion(
    "Thank you for your email.",
    context={"text_type": "email", "formality": "business"},
    style="professional",
    expected_length="sentence"
)

# Creative writing completion
completions = predictor.predict_completion(
    "The old house stood silently",
    context={"text_type": "creative", "genre": "mystery"},
    expected_length="paragraph"
)
```

**Return Value Structure:**
```python
{
    'text': 'the meeting has been rescheduled to next week.',
    'confidence': 0.85,
    'type': 'completion',
    'word_count': 8,
    'source': 'semantic',
    'metadata': {...}
}
```

---

## Utility Methods

### complete()

```python
complete(partial, context="") -> List[str]
```

Complete a partial word with optional context.

**Parameters:**
- `partial` *(str)*: Partial word to complete
- `context` *(str)*: Optional context

**Returns:** List of completions

**Examples:**
```python
# Word completion
predictor.complete("hel")  # ['hello', 'help', 'helped', 'helping', 'helps']

# With context
predictor.complete("gr", "Good ")  # ['great', 'green', 'group']
```

---

### reset_personal_data()

```python
reset_personal_data()
```

Clear all personal learning data from selections.

**Example:**
```python
predictor.reset_personal_data()
```

---

### cleanup_semantic_data()

```python
cleanup_semantic_data(days=90) -> int
```

Clean up old semantic patterns to free storage space.

**Parameters:**
- `days` *(int)*: Remove patterns older than this many days (default: 90)

**Returns:** Number of patterns removed

**Example:**
```python
# Clean up patterns older than 30 days
removed_count = predictor.cleanup_semantic_data(days=30)
print(f"Removed {removed_count} old patterns")
```

---

### reset_semantic_data()

```python
reset_semantic_data()
```

Clear all semantic learning data.

**Example:**
```python
predictor.reset_semantic_data()
```

---

## Properties

### stats

```python
@property
stats -> Dict[str, Any]
```

Get usage statistics including semantic data.

**Returns:** Dictionary with statistics about usage, database size, top words, and semantic patterns

**Example:**
```python
stats = predictor.stats
print(f"Personal selections: {stats['personal_selections']}")
print(f"Database size: {stats['database_size_mb']} MB")
print(f"Top words: {stats['top_words']}")
print(f"Semantic patterns: {stats['semantic']}")
```

**Return Structure:**
```python
{
    'personal_selections': 1250,
    'database_size_mb': 15.3,
    'top_words': [('the', 45), ('and', 32), ('to', 28)],
    'semantic': {
        'total_patterns': 89,
        'total_embeddings': 89,
        'last_updated': '2024-01-15T10:30:00'
    }
}
```

---

### has_semantic

```python
@property
has_semantic -> bool
```

Check if semantic features are available.

**Returns:** Boolean indicating if ChromaDB is installed and working

**Example:**
```python
if predictor.has_semantic:
    completions = predictor.predict_completion("Hello")
else:
    print("Semantic features not available")
```

---

## Configuration Methods

### export_config()

```python
export_config(path)
```

Export current configuration to a file.

**Parameters:**
- `path` *(str)*: Path to save configuration file

**Example:**
```python
predictor.export_config("/path/to/config.json")
```

---

## Callback Methods

### on_prediction()

```python
on_prediction(callback)
```

Set callback for when predictions are made.

**Parameters:**
- `callback` *(callable)*: Function to call on predictions

**Example:**
```python
def log_predictions(text, predictions):
    print(f"Predicted for '{text}': {predictions}")

predictor.on_prediction(log_predictions)
```

---

### on_selection()

```python
on_selection(callback)
```

Set callback for when selections are recorded.

**Parameters:**
- `callback` *(callable)*: Function to call on selections

**Example:**
```python
def log_selections(context, word, index):
    print(f"Selected '{word}' for context '{context}' at index {index}")

predictor.on_selection(log_selections)
```

---

## Convenience Functions

### create_predictor()

```python
create_predictor(**kwargs) -> Predictpy
```

Create a Predictpy instance with optional configuration.

**Parameters:**
- `**kwargs`: Any valid Predictpy constructor arguments

**Returns:** Configured Predictpy instance

**Example:**
```python
from predictpy import create_predictor

# Quick creation with custom settings
predictor = create_predictor(
    training_size="large",
    use_semantic=True,
    db_path="/custom/path.db"
)
```

---

## Error Handling

All methods handle errors gracefully and log warnings for non-critical issues:

```python
try:
    predictor = Predictpy(use_semantic=True)
    suggestions = predictor.predict("Hello")
except Exception as e:
    print(f"Error: {e}")
    # Fallback to basic prediction
    predictor = Predictpy(use_semantic=False)
```

Common error scenarios:
- **ChromaDB not available**: Semantic features disabled, word prediction still works
- **Database corruption**: Automatic regeneration with warning
- **Invalid input**: Empty lists returned with logged warnings
- **Network issues**: Cached results used when available

---

## Type Hints

Predictpy includes full type hints for better IDE support:

```python
from typing import List, Dict, Any, Optional, Union
from predictpy import Predictpy

predictor: Predictpy = Predictpy()
suggestions: List[str] = predictor.predict("Hello")
completions: List[Dict[str, Any]] = predictor.predict_completion("Hello")
```
