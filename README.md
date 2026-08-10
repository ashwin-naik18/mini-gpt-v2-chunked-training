# LLM-v2 — A GPT-style Language Model Built from Scratch

A minimal, educational implementation of a GPT-style transformer language model built from scratch in PyTorch, trained on the [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) dataset.

This project implements the core building blocks of a decoder-only transformer — multi-head self-attention, feed-forward networks, layer normalization, and residual connections — without relying on high-level transformer libraries, as a way of understanding how GPT-style models work under the hood.

## What Changed from v1

**v1** worked, but didn't scale. The entire tokenized dataset was encoded and loaded into memory as a single tensor (`encoded.pt`), and the whole sequence was fed into the model in one forward pass with `block_size = len(encoded)`. This meant:

- Memory usage grew linearly with dataset size, making it impossible to train on anything beyond a tiny slice of data
- There was no batching — a single giant sequence was passed through the model at once instead of many fixed-length training examples
- The model couldn't be trained on GPU at any real scale, since both the data and the attention computation scaled with the full sequence length
- Preprocessing and training were tightly coupled to whatever fit in RAM at once

**v2** fixes this by rethinking the data pipeline for scalability:

| | v1 | v2 |
|---|---|---|
| Data storage | Single `encoded.pt` tensor holding the entire dataset | Dataset split into fixed-size **chunks** (`chunk_0.pt`, `chunk_1.pt`, …), loaded one at a time |
| Training examples | One full-sequence forward pass | Fixed-length windows (`block_size = 128`) sampled via a proper `Dataset`/`DataLoader`, with shuffled mini-batches |
| Context length | `block_size` = length of entire dataset | `block_size` = 128, independent of dataset size |
| Attention masking | Mask recomputed every forward pass | Causal mask precomputed and registered as a buffer |
| Device support | CPU only (implicit) | Runs on GPU when available (`device = "cuda" if torch.cuda.is_available() else "cpu"`) |
| Memory footprint | Entire dataset + full-sequence attention matrix in memory at once | Only one chunk and one batch in memory at a time; chunks are deleted and cache cleared between iterations |
| Checkpointing | None — model was never saved | Best model (by validation loss) is checkpointed to disk during training |
| Storage | Local files, tied to a single machine/session | Google Drive-backed storage (`setup.py`), so preprocessing and training can resume across sessions |

In short, v1 was a proof-of-concept that a GPT-style model could be trained from scratch; v2 turns that into a pipeline that can actually scale to a full dataset without running out of memory.

## Features

- **Custom transformer implementation** — self-attention heads, multi-head attention, feed-forward blocks, and transformer blocks written from scratch (no `nn.Transformer` or HuggingFace models)
- **Causal (masked) self-attention** for autoregressive next-token prediction
- **Word-level tokenizer** built directly from the training corpus, with special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`)
- **Chunked data preprocessing** to handle large datasets without exhausting memory
- **Checkpointing** — best model (by validation loss) is saved during training
- **Text generation** via multinomial sampling

## Project Structure

```
.
├── setup.py         # Mounts Google Drive and defines the save directory (Colab-specific)
├── preprocessor.py  # Downloads SimpleStories, builds vocab, tokenizes and chunks the data
├── train.py          # Defines the GPT model and runs the training loop
└── README.md
```

## Architecture

The model is a decoder-only transformer, similar in spirit to GPT-2:

| Component      | Description                                              |
|-----------------|-----------------------------------------------------------|
| Token Embedding | Maps token ids to dense vectors                          |
| Position Embedding | Learned positional encodings                          |
| Attention Head  | Scaled dot-product attention with a causal mask           |
| Multi-Head Attention | Multiple attention heads concatenated and projected |
| Feed-Forward Network | 2-layer MLP with ReLU, 4x expansion                  |
| Transformer Block | Pre-LayerNorm attention + FFN with residual connections |
| LM Head         | Final linear layer projecting to vocabulary logits         |

### Default hyperparameters

| Parameter       | Value |
|------------------|-------|
| Embedding dimension | 64 |
| Block size (context length) | 128 |
| Number of layers | 4 |
| Number of attention heads | 4 |
| Batch size | 64 |
| Optimizer | Adam (lr = 3e-4) |
| Epochs | 10 |

## Dataset

The model is trained on [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories), a dataset of short, simple children's stories — well suited for training small language models from scratch with limited compute.

Text is tokenized at the **word level** (whitespace-split), and each story is wrapped with `<BOS>` / `<EOS>` tokens before being encoded into a single flat token stream and split into fixed-size chunks for efficient storage and loading.

## Setup

This project was built to run on **Google Colab** with data stored on Google Drive.

### 1. Clone the repository

```bash
git clone https://github.com/ashwin-naik18/llm-v2.git
cd mini-gpt-v2-chunked-training
```

### 2. Install dependencies

```bash
pip install torch datasets
```

### 3. Configure storage

`setup.py` mounts Google Drive and defines `save_dir`, the directory where vocab, config, data chunks, and checkpoints are stored. Update `save_dir` if you want to use local storage instead of Google Drive, or if you are not running in Colab.

## Usage

### 1. Preprocess the data

Downloads the dataset, builds the vocabulary, tokenizes all stories, and saves them as chunked tensors:

```bash
python preprocessor.py
```

This produces, inside `save_dir`:
- `vocab.pt` — the vocabulary list
- `config.pt` — dataset config (vocab size, number of chunks)
- `chunk_0.pt`, `chunk_1.pt`, … — tokenized data chunks

### 2. Train the model

```bash
python train.py
```

Training iterates over each data chunk, splits it into train/validation sets (90/10), and trains for a fixed number of epochs. The best model (lowest validation loss) is checkpointed to `best_model.pth`, which contains the model weights, vocabulary, and architecture config needed to reload the model later.
