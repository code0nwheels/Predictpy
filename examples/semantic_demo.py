"""
Enhanced example demonstrating Predictpy's semantic completion features.
This example shows how to use both traditional word prediction and the new semantic completion.
"""

from predictpy import Predictpy
import json


def main():
    """Demonstrate semantic completion features."""
    print("Predictpy Semantic Completion Demo")
    print("=" * 50)
      # Initialize with semantic features enabled
    predictor = Predictpy(use_semantic=True, training_size="small")
    print("Predictpy initialized with semantic features")
    
    # Check if semantic features are available
    print(f"Semantic features available: {predictor.has_semantic}")
    
    # Sample training data - different types of text
    training_texts = {
        "email": [
            "Thank you for your email. I wanted to let you know that the meeting has been rescheduled to next Tuesday.",
            "I appreciate your quick response. The project is progressing well and we should have the deliverables ready by Friday.",
            "Following up on our conversation yesterday, I've attached the requested documents for your review.",
        ],
        "chat": [
            "Hey, how are you doing today? I was thinking we could grab lunch later.",
            "That sounds great! Let me know what time works best for you.",
            "Perfect, see you at noon. Looking forward to catching up.",
        ],
        "document": [
            "The main advantages of this approach are efficiency, scalability, and maintainability.",
            "In conclusion, the implementation of these features will significantly improve user experience.",
            "The results demonstrate a clear improvement in performance metrics across all test scenarios.",
        ]
    }
      # Learn from different types of text
    print("\nLearning from sample texts...")
    for text_type, texts in training_texts.items():
        for text in texts:
            predictor.learn_from_text(text, text_type=text_type)
        print(f"  Learned {len(texts)} {text_type} samples")
    
    # Demonstrate traditional word prediction
    print("\nTraditional Word Prediction:")
    traditional_tests = [
        "I want to",
        "Thank you for",
        "The meeting has",
    ]
    
    for test_text in traditional_tests:
        predictions = predictor.predict(test_text, count=3)
        print(f"  '{test_text}' → {predictions}")
    
    # Demonstrate semantic completion (if available)
    if predictor.has_semantic:
        print("\nSemantic Thought Completion:")
        completion_tests = [
            "I wanted to let you know that",
            "Thank you for your email.",
            "The main advantages are",
            "In conclusion,",
        ]
        
        for test_text in completion_tests:
            completions = predictor.predict_completion(test_text, min_words=3)
            print(f"\n  '{test_text}'")
            for i, completion in enumerate(completions[:3], 1):
                confidence = completion.get('confidence', 0)
                completion_text = completion.get('text', '')
                print(f"    {i}. {completion_text} (confidence: {confidence:.2f})")
        
        # Demonstrate context-aware completion
        print("\nContext-Aware Completion (Email style):")
        email_completion = predictor.predict_completion(
            "Thanks for reaching out.",
            context={"text_type": "email"},
            min_words=5
        )
        for completion in email_completion[:2]:
            print(f"  → {completion['text']} (confidence: {completion['confidence']:.2f})")
        
        print("\nContext-Aware Completion (Chat style):")
        chat_completion = predictor.predict_completion(
            "Hey, how are you",
            context={"text_type": "chat"},
            min_words=3
        )
        for completion in chat_completion[:2]:
            print(f"  → {completion['text']} (confidence: {completion['confidence']:.2f})")
    
    # Show statistics
    print("\nPredictpy Statistics:")
    stats = predictor.stats
    print(json.dumps(stats, indent=2, default=str))
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit):")
    print("Enter partial text for completion suggestions...")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Try both traditional and semantic predictions
            print(f"\nWord predictions: {predictor.predict(user_input, count=8)}")
            
            if predictor.has_semantic:
                completions = predictor.predict_completion(user_input, min_words=2)
                if completions:
                    print("Semantic completions:")
                    for i, completion in enumerate(completions[:3], 1):
                        print(f"  {i}. {completion['text']} (confidence: {completion['confidence']:.2f})")
                else:
                    print("No semantic completions found")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThanks for trying Predictpy!")


if __name__ == "__main__":
    main()
