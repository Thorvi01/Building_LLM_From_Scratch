import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


with open("work_with_text_data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257     # GPT-2's BPE vocabulary size
output_dim = 256       # size of each embedding vector
context_length = 4     # tokens per training example
batch_size = 8

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

dataloader = create_dataloader_v1(
    raw_text, batch_size=batch_size, max_length=context_length,
    stride=context_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs shape:", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Token embeddings shape:", token_embeddings.shape)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Positional embeddings shape:", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("Final input embeddings shape:", input_embeddings.shape)