import os
import requests

def download_the_verdict(file_path="the-verdict.txt"):
    if not os.path.exists(file_path):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        response = requests.get(url, timeout=30)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

raw_text = download_the_verdict("work_with_text_data/the-verdict.txt")
print("Total number of characters:", len(raw_text))
print(raw_text[:99])

import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print("Number of tokens:", len(preprocessed))
print("First 10 tokens:", preprocessed[:10])

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print("New vocabulary size:", len(vocab))

vocab = {token: integer for integer, token in enumerate(all_tokens)}

# Let's peek at the first 10 entries in our vocabulary
for i, (token, integer) in enumerate(vocab.items()):
    print(token, "->", integer)
    if i >= 9:
        break

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizer(vocab)

tokenizer = SimpleTokenizer(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print("Combined text:", text)
print("Encoded:", tokenizer.encode(text))
print("Decoded:", tokenizer.decode(tokenizer.encode(text)))   