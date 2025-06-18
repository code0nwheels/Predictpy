# Getting Started

This guide will help you get up and running with Predictpy quickly.

## Installation

### Basic Installation

Install Predictpy from PyPI using pip:

```bash
pip install predictpy
```

### From Source

For the latest development version:

```bash
git clone https://github.com/code0nwheels/Predictpy.git
cd Predictpy
pip install -e .
```

### Requirements

- Python 3.7 or higher
- Required packages (installed automatically):
  - `nltk >= 3.6.0`
  - `datasets >= 2.0.0` 
  - `chromadb >= 0.4.0`
  - `sentence-transformers >= 2.2.0`

## Basic Usage

### Simple Word Prediction

```python
from predictpy import Predictpy

# Initialize the predictor
predictor = Predictpy()

# Get word predictions
suggestions = predictor.predict("I want to")
print(suggestions)  # ['go', 'be', 'see', 'make', 'do']

# Get predictions with partial word
suggestions = predictor.predict("I want to g")
print(suggestions)  # ['go', 'get', 'give', 'good', 'going']
```

### Learning from User Selections

```python
# Record user selections to improve future predictions
predictor.select("I want", "go")

# The predictor learns and will prioritize "go" next time
improved_suggestions = predictor.predict("I want")
print(improved_suggestions)  # ['go', 'to', 'be', 'see', 'make']
```

### Text Completion

```python
# Enable semantic features for thought completion
predictor = Predictpy(use_semantic=True)

# Learn from your writing style
predictor.learn_from_text("""
Thank you for your email. I wanted to let you know that 
the meeting has been rescheduled to next Tuesday.
""", text_type="email")

# Get intelligent completions
completions = predictor.predict_completion("Thank you for your")
for completion in completions:
    print(f"→ {completion['text']} (confidence: {completion['confidence']:.2f})")
```

## First Steps

### 1. Initialize Predictpy

```python
from predictpy import Predictpy

# Basic initialization (automatic training)
predictor = Predictpy()

# Custom configuration
predictor = Predictpy(
    training_size="medium",  # small, medium, or large
    use_semantic=True,       # enable AI completions
    auto_train=True          # train on first use
)
```

### 2. Make Your First Predictions

```python
# String input (most common)
suggestions = predictor.predict("Hello w")
print(suggestions)  # ['world', 'will', 'was', 'would', 'with']

# List input for more control
suggestions = predictor.predict(["Hello"], count=3)
print(suggestions)  # ['world', 'there', 'everyone']
```

### 3. Improve with Learning

```python
# Method 1: Record selections
predictor.select("Hello", "world")

# Method 2: Learn from text blocks
predictor.learn_from_text("""
Hello world! This is a sample text that the predictor
can learn from to improve future predictions.
""")
```

## Configuration Options

### Training Size

Choose training size based on your needs:

```python
# Fast startup, basic predictions (1k sentences)
predictor = Predictpy(training_size="small")

# Balanced performance (10k sentences) - default
predictor = Predictpy(training_size="medium")

# Best predictions, slower startup (50k sentences)
predictor = Predictpy(training_size="large")
```

### Database Location

```python
# Custom database path
predictor = Predictpy(db_path="/path/to/custom.db")

# Use existing database without training
predictor = Predictpy(auto_train=False)
```

### Semantic Features

```python
# Enable semantic completion (requires ChromaDB)
predictor = Predictpy(use_semantic=True)

# Disable semantic features for word-only prediction
predictor = Predictpy(use_semantic=False)
```

## Interactive Example

Here's a complete example showing interactive usage:

```python
from predictpy import Predictpy

# Initialize
predictor = Predictpy()

# Simulate typing with predictions
text = "I am"
while len(text.split()) < 10:  # Build a 10-word sentence
    # Get predictions for current text
    suggestions = predictor.predict(text, count=3)
    print(f"Text: '{text}'")
    print(f"Suggestions: {suggestions}")
    
    # Simulate user selecting first suggestion
    selected_word = suggestions[0] if suggestions else "happy"
    
    # Record the selection for learning
    last_words = text.split()[-2:] if len(text.split()) >= 2 else text.split()
    predictor.select(last_words, selected_word)
    
    # Add to text
    text += " " + selected_word

print(f"Final text: {text}")
```

## Next Steps

- Explore the [API Reference](api-reference.md) for complete method documentation
- Check out [Examples](examples.md) for real-world usage patterns
- Learn about [Semantic Features](semantic-features.md) for advanced text completion
- Review [Configuration](configuration.md) for detailed setup options
