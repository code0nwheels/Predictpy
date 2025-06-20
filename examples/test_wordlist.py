"""
Quick test of the WordList functionality.
"""
from predictpy.wordlist import WordList

def main():
    print("Testing WordList functionality...")
    
    # Test the comprehensive list
    word_list = WordList('comprehensive')
    words = word_list.load_words()
    print(f"Loaded comprehensive list with {len(words):,} words")
    print(f"Is 'python' a valid word? {word_list.is_valid_word('python')}")
    print(f"Is 'javascript' a valid word? {word_list.is_valid_word('javascript')}")
    
    # Test the common list
    common_list = WordList('common')
    common_words = common_list.load_words()
    print(f"Loaded common list with {len(common_words):,} words")
    print(f"Is 'the' in common words? {common_list.is_valid_word('the')}")
    
    print("Test completed!")

if __name__ == "__main__":
    main()
