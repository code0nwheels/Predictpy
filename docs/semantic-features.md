# Semantic Features

Predictpy includes advanced AI-powered semantic completion that learns your writing patterns and can complete thoughts, sentences, and paragraphs intelligently.

## Overview

### Traditional vs Semantic Prediction

| Feature | Traditional (Word-level) | Semantic (Thought-level) |
|---------|-------------------------|-------------------------|
| **Scope** | Next word only | Complete thoughts/sentences |
| **Context** | Last 2-3 words | Full semantic meaning |
| **Learning** | Word frequency patterns | Writing style & context |
| **Technology** | N-gram models | Neural embeddings (ChromaDB) |
| **Use Case** | Autocomplete, word suggestions | Content generation, thought completion |
| **Speed** | ~10ms | ~50-200ms (cached ~10ms) |

### When to Use Semantic Features

**Best for:**
- Content writing and composition
- Email and message completion
- Creative writing assistance
- Long-form text generation
- Context-aware suggestions

**Traditional prediction better for:**
- Real-time typing assistance
- Simple word completion
- Performance-critical applications
- Minimal resource usage

---

## Setup and Configuration

### Enable Semantic Features

```python
from predictpy import Predictpy

# Enable semantic features (default)
predictor = Predictpy(use_semantic=True)

# Check if semantic features are available
if predictor.has_semantic:
    print("✓ Semantic features ready")
else:
    print("✗ ChromaDB not available")
```

### Requirements

Semantic features require additional dependencies:
- `chromadb >= 0.4.0`
- `sentence-transformers >= 2.2.0`

These are installed automatically with Predictpy, but you can install manually:

```bash
pip install chromadb sentence-transformers
```

### Storage Configuration

```python
# Custom semantic database location
predictor = Predictpy(
    use_semantic=True,
    db_path="/custom/path/predictpy.db"  # Semantic data stored in /custom/path/chroma/
)
```

---

## Core Semantic Methods

### Learning from Text

The semantic system learns from complete thoughts and writing patterns:

```python
# Learn from email content
predictor.learn_from_text("""
Thank you for your email regarding the project update. 
I wanted to let you know that we're making excellent progress 
and expect to deliver on schedule. Please let me know if you 
have any questions or concerns.
""", text_type="email", tags=["business", "project"])

# Learn from chat messages
predictor.learn_from_text("""
Hey! How's it going? I hope you're having a great day. 
Did you manage to finish that report we talked about?
""", text_type="chat", tags=["casual", "check-in"])

# Learn from creative writing
predictor.learn_from_text("""
The old lighthouse stood silently against the stormy sky, 
its beacon cutting through the darkness like a sword of light. 
Waves crashed against the rocky shore below, sending spray 
high into the air.
""", text_type="creative", tags=["description", "atmosphere"])
```

### Intelligent Completion

```python
# Basic thought completion
completions = predictor.predict_completion("I wanted to let you know that")

for completion in completions:
    print(f"→ {completion['text']}")
    print(f"  Confidence: {completion['confidence']:.2f}")
    print(f"  Type: {completion['type']}")
    print()

# Output:
# → the meeting has been rescheduled to next Tuesday.
#   Confidence: 0.85
#   Type: sentence_completion
#
# → we've made significant progress on the project.
#   Confidence: 0.78
#   Type: sentence_completion
```

---

## Context-Aware Completion

### Text Type Context

```python
# Email completion
email_completions = predictor.predict_completion(
    "Thank you for your",
    context={"text_type": "email", "formality": "business"}
)

# Chat completion  
chat_completions = predictor.predict_completion(
    "Thanks for",
    context={"text_type": "chat", "formality": "casual"}
)

# Creative writing completion
creative_completions = predictor.predict_completion(
    "The ancient castle",
    context={"text_type": "creative", "genre": "fantasy"}
)
```

### Style and Tone

```python
# Formal style
formal_completion = predictor.predict_completion(
    "I would like to inform you that",
    style="formal",
    context={"formality": "high", "tone": "professional"}
)

# Casual style
casual_completion = predictor.predict_completion(
    "Hey, just wanted to say that",
    style="casual", 
    context={"formality": "low", "tone": "friendly"}
)

# Academic style
academic_completion = predictor.predict_completion(
    "The research findings indicate that",
    style="academic",
    context={"formality": "high", "tone": "objective"}
)
```

### Length Control

```python
# Short completion (sentence)
short_completion = predictor.predict_completion(
    "The weather today is",
    expected_length="sentence",
    min_words=3
)

# Medium completion (multiple sentences)
medium_completion = predictor.predict_completion(
    "I've been thinking about our conversation",
    expected_length="paragraph",
    min_words=15
)

# Long completion (extended thought)
long_completion = predictor.predict_completion(
    "The implications of this discovery",
    expected_length="extended",
    min_words=25
)
```

---

## Advanced Usage Patterns

### Domain-Specific Learning

```python
# Technical documentation
predictor.learn_from_text("""
To configure the API endpoint, first ensure that your 
authentication credentials are properly set. Then, 
initialize the client with your API key and base URL.
""", text_type="technical", tags=["documentation", "api"])

# Marketing copy
predictor.learn_from_text("""
Transform your workflow with our innovative solution. 
Boost productivity by 300% while reducing costs and 
improving team collaboration across all departments.
""", text_type="marketing", tags=["sales", "benefits"])

# Support responses
predictor.learn_from_text("""
I understand your frustration with this issue. Let me 
help you resolve this quickly. Can you please provide 
the error message you're seeing?
""", text_type="support", tags=["customer-service", "empathy"])
```

### Multi-Context Completion

```python
# Context-sensitive email reply
completions = predictor.predict_completion(
    "Thank you for bringing this to my attention.",
    context={
        "text_type": "email",
        "conversation_type": "issue_response", 
        "formality": "business",
        "tone": "concerned",
        "next_action": "investigation"
    }
)

# Meeting follow-up
completions = predictor.predict_completion(
    "Following up on our discussion about",
    context={
        "text_type": "email",
        "conversation_type": "meeting_followup",
        "formality": "business", 
        "tone": "collaborative",
        "timeline": "actionable"
    }
)
```

### Iterative Completion

```python
# Build longer content iteratively
base_text = "The new product launch strategy"

# First completion
first_completion = predictor.predict_completion(
    base_text,
    context={"text_type": "business", "stage": "introduction"}
)

# Extend with second completion
extended_text = base_text + " " + first_completion[0]['text']
second_completion = predictor.predict_completion(
    extended_text,
    context={"text_type": "business", "stage": "details"}
)

# Final extended content
final_text = extended_text + " " + second_completion[0]['text']
print(final_text)
```

---

## Performance and Storage

### Storage Management

```python
# Check semantic storage usage
stats = predictor.stats
semantic_info = stats['semantic']
print(f"Total patterns: {semantic_info['total_patterns']}")
print(f"Total embeddings: {semantic_info['total_embeddings']}")
print(f"Last updated: {semantic_info['last_updated']}")

# Clean up old patterns (older than 30 days)
removed = predictor.cleanup_semantic_data(days=30)
print(f"Removed {removed} old patterns")

# Reset all semantic data
predictor.reset_semantic_data()
```

### Performance Optimization

```python
# Warm up the semantic model (first completion is slower)
predictor.predict_completion("Hello", min_words=1)

# Batch learning for better performance
texts = [
    "First document content...",
    "Second document content...",
    "Third document content..."
]

for text in texts:
    predictor.learn_from_text(text, text_type="batch_learning")
```

### Memory Usage

- **Embedding model**: ~90MB (sentence-transformers model)
- **Per pattern**: ~1KB (text + embedding + metadata)
- **Database overhead**: ~10MB (ChromaDB)
- **Typical usage**: 50-200MB for 1000 learned patterns

---

## Best Practices

### Learning Strategy

```python
# 1. Start with diverse, high-quality examples
predictor.learn_from_text(your_best_writing_samples, text_type="reference")

# 2. Learn incrementally from real usage
def on_user_completion(context, selected_completion):
    # User approved this completion, learn from it
    predictor.learn_from_text(
        context + " " + selected_completion,
        text_type="approved_completion"
    )

# 3. Use specific tags for better retrieval
predictor.learn_from_text(
    text,
    text_type="email",
    tags=["customer_service", "billing", "resolved"]
)
```

### Context Design

```python
# Good: Specific, relevant context
context = {
    "text_type": "email",
    "recipient": "client", 
    "purpose": "status_update",
    "formality": "business",
    "tone": "reassuring"
}

# Avoid: Vague or conflicting context
context = {
    "text_type": "everything",
    "tone": "formal_and_casual",  # conflicting
    "random_field": "random_value"  # irrelevant
}
```

### Error Handling

```python
try:
    completions = predictor.predict_completion(text)
    if not completions:
        # Fallback to word prediction
        fallback = predictor.predict(text.split()[-2:])
        # Create completion-like structure
        completions = [{"text": " ".join(fallback), "confidence": 0.5}]
except Exception as e:
    print(f"Semantic completion failed: {e}")
    # Use word prediction as backup
    completions = []
```

---

## Integration Examples

### Text Editor Integration

```python
class SmartTextEditor:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True)
        self.current_text = ""
    
    def get_suggestions(self, cursor_position):
        # Get current context
        context = self.current_text[:cursor_position]
        
        # Try semantic completion for longer contexts
        if len(context.split()) >= 3:
            completions = self.predictor.predict_completion(
                context,
                min_words=3,
                context={"text_type": "document"}
            )
            if completions:
                return [c['text'] for c in completions[:3]]
        
        # Fallback to word prediction
        return self.predictor.predict(context, count=5)

    def on_text_accepted(self, original_context, selected_text):
        # Learn from user acceptance
        self.predictor.learn_from_text(
            original_context + " " + selected_text,
            text_type="user_document"
        )
```

### Chat Application Integration

```python
class SmartChatBot:
    def __init__(self):
        self.predictor = Predictpy(use_semantic=True)
        
        # Pre-train on chat patterns
        self.predictor.learn_from_text("""
        Hey! How are you doing?
        Thanks for reaching out.
        I'll get back to you soon.
        Let me know if you need anything else.
        """, text_type="chat", tags=["responses"])
    
    def suggest_replies(self, incoming_message):
        # Context-aware reply suggestions
        if "how are you" in incoming_message.lower():
            context = {"text_type": "chat", "response_to": "greeting"}
        elif "thank" in incoming_message.lower():
            context = {"text_type": "chat", "response_to": "gratitude"}
        else:
            context = {"text_type": "chat", "response_to": "general"}
        
        # Generate contextual completions
        replies = self.predictor.predict_completion(
            "Thanks for your message.",  # starter
            context=context,
            style="friendly"
        )
        
        return [reply['text'] for reply in replies[:3]]
```

---

## Troubleshooting

### Common Issues

**Semantic features not available:**
```python
# Check availability
if not predictor.has_semantic:
    print("Installing ChromaDB...")
    import subprocess
    subprocess.run(["pip", "install", "chromadb", "sentence-transformers"])
```

**Slow first completion:**
```python
# Warm up the model during initialization
predictor = Predictpy(use_semantic=True)
# First completion loads the model (slow)
predictor.predict_completion("warm up", min_words=1)
# Subsequent completions are fast
```

**Memory usage too high:**
```python
# Clean up regularly
if predictor.stats['semantic']['total_patterns'] > 1000:
    removed = predictor.cleanup_semantic_data(days=60)
    print(f"Cleaned up {removed} old patterns")
```

**Poor completion quality:**
```python
# Need more diverse training data
training_texts = [
    "High-quality example 1...",
    "High-quality example 2...",
    "High-quality example 3..."
]

for text in training_texts:
    predictor.learn_from_text(text, text_type="quality_training")
```
