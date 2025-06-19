"""
Interactive CLI example of Predictpy with keystroke-based predictions
"""

import os
import sys
import msvcrt
from predictpy import WordPredictionEngine

def get_key():
    """Get a single keypress from the user."""
    ch = msvcrt.getch()
    if ch in (b'\x00', b'\xe0'):  # Special key prefix
        msvcrt.getch()  # Read and discard the second byte
        return None
    try:
        return ch.decode('utf-8')
    except UnicodeDecodeError:
        return chr(ord(ch[0]))

def clear_line():
    """Clear the current line in the terminal."""
    sys.stdout.write('\r' + ' ' * 100 + '\r')
    sys.stdout.flush()
    
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls')

def main():
    """Interactive word prediction example with keystroke-based predictions."""    # Initialize the engine (will train automatically if needed)
    print("Initializing Predictpy...")
    engine = WordPredictionEngine(auto_train=True, target_sentences=5000000)
    clear_screen()
    print("Initialization complete.")

    # Get and print vocab count
    vocab_count = engine.get_vocab_count()
    print(f"Vocabulary size: {vocab_count} unique words")
    
    # Number of suggestions to show (can be customized)
    num_suggestions = 8  # Change this value to get more or fewer suggestions
    
    print("\nType a sentence. Press Tab to accept the first suggestion or press 1-8 to select a specific suggestion.")
    print("Press Esc to exit. Suggestions will appear as you type.\n")
    
    # Initialize state
    current_text = []
    current_word = ""
    predictions = []
    context = []

    # Main interaction loop
    while True:
        # Display the current text and predictions
        clear_screen()
        print("Interactive Predictpy Demo - Tab for 1st suggestion or keys 1-8 to select, Esc to exit\n")
        
        displayed_text = " ".join(current_text) + (" " if current_text else "") + current_word
        print(f"Current text: {displayed_text}")
        
        # Show predictions if we have any
        print("\nSuggestions:")
        if predictions:
            prediction_text = " | ".join([f"{i+1}: {pred}" for i, pred in enumerate(predictions[:num_suggestions])])
            print(prediction_text)
        else:
            print("(Type to see predictions)")
        
        # Get input
        key = get_key()
        
        # Skip if key is None (special key)
        if key is None:
            continue

        # Handle escape key
        if key == '\x1b' or ord(key) == 27:  # ESC
            print("\n\nExiting...")
            break
            
        # Handle tab key for selection
        elif key == '\t' or ord(key) == 9:  # TAB
            if predictions:
                # Select the first prediction
                selected_word = predictions[0]
                if current_word:
                    # Remove the partial word and add the full prediction
                    current_text.append(selected_word)
                else:
                    # Just add the prediction
                    current_text.append(selected_word)
                
                # Record the selection for learning
                engine.record_selection(context, selected_word)
                
                # Reset current word and update context
                current_word = ""
                context = current_text[-2:] if len(current_text) >= 2 else current_text[:]
                predictions = engine.predict(context, current_word, num_suggestions)
                
        # Handle number keys for selecting specific predictions
        elif key in "12345678" and predictions:
            try:
                index = int(key) - 1
                if index < len(predictions):
                    selected_word = predictions[index]
                    if current_word:
                        # Remove the partial word and add the selected prediction
                        current_text.append(selected_word)
                    else:
                        # Just add the selected prediction
                        current_text.append(selected_word)
                    
                    # Record the selection for learning
                    engine.record_selection(context, selected_word)
                    
                    # Reset current word and update context
                    current_word = ""
                    context = current_text[-2:] if len(current_text) >= 2 else current_text[:]
                    predictions = engine.predict(context, current_word, num_suggestions)
                    
            except (ValueError, IndexError):
                pass
        
        # Handle space key
        elif key == " ":
            if current_word:
                current_text.append(current_word)
                current_word = ""
                # Update context with last two words
                context = current_text[-2:] if len(current_text) >= 2 else current_text[:]
                predictions = engine.predict(context, current_word, num_suggestions)
                
        # Handle backspace
        elif key == '\b' or ord(key) == 8:  # Windows backspace
            if current_word:
                current_word = current_word[:-1]
            elif current_text:
                current_word = current_text.pop()
                current_word = current_word[:-1]
            
            # Update context
            context = current_text[-2:] if len(current_text) >= 2 else current_text[:]
            predictions = engine.predict(context, current_word, num_suggestions)
            
        # Handle regular text input
        elif key.isprintable():
            current_word += key
            # Get predictions based on context and current partial word
            context = current_text[-2:] if len(current_text) >= 2 else current_text[:]
            predictions = engine.predict(context, current_word, num_suggestions)
            
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
