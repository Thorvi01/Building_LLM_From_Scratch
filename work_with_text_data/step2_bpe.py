import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Token IDs:", integers)

strings = tokenizer.decode(integers)
print("Decoded:", strings)

print("Vocabulary size:", tokenizer.n_vocab)

sample_text = "In the sunlit terraces of someunknownPlace, the crowd gathered slowly."
enc_text = tokenizer.encode(sample_text)
print("Total tokens in this sentence:", len(enc_text))

context_size = 4
x = enc_text[:context_size]
y = enc_text[1:context_size + 1]

print("Input:", x)
print("Target:", y)

# Let's see it as an actual prediction task, one token at a time
for i in range(1, context_size + 1):
    context = enc_text[:i]
    desired = enc_text[i]
    print(context, "---->", desired)