import torch
from datasets import load_dataset
import torch

print("Loading dataset...")
dataset = load_dataset("SimpleStories/SimpleStories")

print("Dataset Loaded..")

stories = dataset['train']['story']

vocab = set()

for story in stories:
  word = story.split()
  vocab.update(word)


special_token = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>"
]

vocab = special_token + sorted(vocab)

stoi = {
    word: i
    for i, word in enumerate(vocab)
}

itos = {
    i : word
    for word, i in stoi.items()
}

print(len(stories))
print(len(vocab))



encoded_chunk = []
chunk_size = 10000
chunk_id = 0

print("Saving Started..")
for i, story in enumerate(stories):
  tokens = ['<BOS>'] + story.split() + ['<EOS>']

  encoded = [
      stoi.get(word, stoi['<UNK>'])
      for word in tokens
  ]

  encoded_chunk.extend(encoded)

  if (i + 1) % chunk_size == 0:
    torch.save(
        torch.tensor(encoded_chunk, dtype=torch.long),
        f"{save_dir}/chunk_{chunk_id}.pt"
    )
    encoded_chunk.clear()
    chunk_id += 1
    

if len(encoded_chunk) > 0:
  torch.save(
      torch.tensor(encoded_chunk, dtype=torch.long),
      f"{save_dir}/chunk_{chunk_id}.pt"
  )
  encoded_chunk.clear()
  
torch.save(vocab, f"{save_dir}/vocab.pt")

config = {
    "vocab_size": len(vocab),
    "num_chunks": chunk_id + 1,
}

torch.save(config, f"{save_dir}/config.pt")

print("Everything Saved..")