class CharacterTokenizer:
    """Convert text to number and back"""

    def __init__(self, text):
        # Get all unique characters

        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {"".join(self.chars[:50])}...")

    def encode(self, text):
        """Convert text to list of integers"""
        return [self.char2idx[c] for c in text]
    
    def decode(self, indices):
        """Convert list of integers back to text"""
        return "".join([self.idx2char[i] for i in indices])
    
# Test
with open("shakespeare.txt", "r") as f:
    data = f.read()

tokenizer = CharacterTokenizer(data)

# encode "Hellp"
encoded = tokenizer.encode("Hello")
print(f"Encoded 'Hello': {encoded}")

# decode back
decoded = tokenizer.decode(encoded)
print(f"Decoded back: {decoded}")

