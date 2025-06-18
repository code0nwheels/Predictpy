"""
Quick Start Guide for Predictpy Semantic Features
"""

from predictpy import Predictpy

# Basic usage - semantic features enabled by default
predictor = Predictpy()

# Learn from your writing
predictor.learn_from_text("""
Thank you for your email. I wanted to let you know that the meeting 
has been rescheduled to next Tuesday at 2 PM. Please let me know if 
this works for your schedule.
""", text_type="email")

predictor.learn_from_text("""
Hey, how's it going? I was thinking we could grab lunch later today. 
Let me know what time works best for you!
""", text_type="chat")

# Traditional word prediction (existing functionality)
print("Word predictions for 'I want to':")
words = predictor.predict("I want to", count=5)
print(words)

# New semantic completion
print("\nSemantic completion for 'Thank you for your':")
completions = predictor.predict_completion("Thank you for your", min_words=3)
for completion in completions:
    print(f"→ {completion['text']} (confidence: {completion['confidence']:.2f})")

# Context-aware completion
print("\nEmail-style completion:")
email_completions = predictor.predict_completion(
    "I wanted to let you know that",
    context={"text_type": "email"}
)
for completion in email_completions[:3]:
    print(f"→ {completion['text']}")

print("\nChat-style completion:")
chat_completions = predictor.predict_completion(
    "Hey, just wanted to say",
    context={"text_type": "chat"}
)
for completion in chat_completions[:3]:
    print(f"→ {completion['text']}")

# Check stats
print(f"\nSemantic features available: {predictor.has_semantic}")
print("Statistics:", predictor.stats)
