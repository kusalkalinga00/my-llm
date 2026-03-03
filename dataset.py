import numpy as np
from tokenizer import CharacterTokenizer


class CharDataset:
    """Create training examples from text data"""

    def __init__(self, text, seq_length=100):

        self.seq_length = seq_length
        self.tokenizer = CharacterTokenizer(text)

        # encode the entire text as integers
        self.data = np.array(self.tokenizer.encode(text))

        # Split into train/val (90% train, 10% val)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        print(f"Train data length: {len(self.train_data)}")
        print(f"Val data length: {len(self.val_data)}")

    def get_batch(self, split="train"):
        """Generate a random batch of sequences"""
        data = self.train_data if split == "train" else self.val_data

        # Random starting positions
        starts = np.random.randint(0, len(data) - self.seq_length, size=32)

        # input: positions [start, start+1, ..., start+seq_length]
        # output : positions [start+1, start+2, ..., start+seq_length+1] (shifted by 1)

        x = np.array([data[s : s + self.seq_length] for s in starts])
        y = np.array([data[s + 1 : s + self.seq_length + 1] for s in starts])

        return x, y


# Test

with open("shakespeare.txt", "r") as f:
    text = f.read()

dataset = CharDataset(text, seq_length=100)

x, y = dataset.get_batch("train")
print(f"Input batch shape: {x.shape}")
print(f"Output batch shape: {y.shape}")
print(f"First input sequence (as chars):\n{dataset.tokenizer.decode(x[0][:20])}")
print(f"First output sequence (as chars):\n{dataset.tokenizer.decode(y[0][:20])}")
