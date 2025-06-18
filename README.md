# Predictpy

A Python package for intelligent word prediction and semantic text completion with personal learning capability.

## Features

- **Smart Word Prediction**: Context-aware suggestions using n-gram language models
- **AI-Powered Completion**: Semantic sentence and paragraph completion
- **Personal Learning**: Adapts to your writing style and preferences  
- **Easy Integration**: Simple API that works with strings or word lists
- **Fast Performance**: Efficient SQLite storage with optional semantic embeddings
- **Configurable**: Flexible setup for different use cases

## Installation

```bash
pip install predictpy
```

## Quick Start

### Basic Word Prediction
```python
from predictpy import Predictpy

# Initialize predictor
predictor = Predictpy()

# Get word predictions
suggestions = predictor.predict("I want to")
print(suggestions)  # ['go', 'be', 'see', 'make', 'do']

# Learn from user selections
predictor.select("I want", "go")
```

### AI Text Completion
```python
# Initialize with semantic features
predictor = Predictpy(use_semantic=True)

# Learn from your writing
predictor.learn_from_text("""
Thank you for your email. I wanted to let you know that 
the meeting has been rescheduled to next Tuesday.
""", text_type="email")

# Get intelligent completions
completions = predictor.predict_completion("Thank you for your")
for completion in completions:
    print(f"→ {completion['text']} (confidence: {completion['confidence']:.2f})")
```

## Configuration

```python
# Different training sizes
predictor = Predictpy(training_size="small")    # Fast startup
predictor = Predictpy(training_size="medium")   # Balanced (default)
predictor = Predictpy(training_size="large")    # Best accuracy

# Custom database path
predictor = Predictpy(db_path="/custom/path.db")

# Disable semantic features for speed
predictor = Predictpy(use_semantic=False)
```

## Documentation

- **[Getting Started](docs/getting-started.md)** - Installation and basic usage
- **[API Reference](docs/api-reference.md)** - Complete method documentation  
- **[Semantic Features](docs/semantic-features.md)** - AI-powered text completion
- **[Configuration](docs/configuration.md)** - Advanced setup options
- **[Examples](docs/examples.md)** - Real-world usage patterns
- **[Advanced Usage](docs/advanced-usage.md)** - Power user features
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## Use Cases

- **Text Editors**: Autocomplete and suggestion systems
- **Chat Applications**: Smart reply suggestions  
- **Writing Assistants**: Content completion and enhancement
- **Accessibility Tools**: Assistive typing interfaces
- **Data Entry**: Speed up repetitive text input

## Requirements

- Python 3.7+
- `nltk >= 3.6.0`
- `datasets >= 2.0.0`
- `chromadb >= 0.4.0` (for semantic features)
- `sentence-transformers >= 2.2.0` (for semantic features)

## License

MIT License - see LICENSE file for details.