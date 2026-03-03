import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt", "r") as f:
    data = f.read()

print(f"Dataset length: {len(data)} characters")
print(f"First 200 characters:\n{data[:200]}")
