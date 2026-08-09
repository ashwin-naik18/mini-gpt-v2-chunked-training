import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from setup import *


device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        super().__init__()

        self.data = data
        self.block_size = block_size


    def __len__(self):
        return len(self.data) - self.block_size


    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]

        y = self.data[idx + 1 : idx + self.block_size + 1]

        return x, y


class Head(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()
        
        
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )
        
        
    def forward(self, x):
        B, T, C = x.shape
        
            
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        score = q @ k.transpose(-2, -1)
        
        score = score / (k.shape[-1] ** 0.5)
        
        mask = self.mask[:T, :T]
        
        score = score.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(score, dim=-1)
        
        output = weights @ v
        
        return output
    
    
class MultiHead(nn.Module):
    def __init__(self, num_head, head_size, embedding_dim, block_size) :
        super().__init__()
        
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_size, block_size) for _ in range(num_head)]
        )
        
        self.proj = nn.Linear(num_head * head_size, embedding_dim)
        
    def forward(self, x):
        output = [head(x) for head in self.heads]
        
        output = torch.cat(output, dim=-1) 
        
        output = self.proj(output)
        
        return output       
    

class FFN(nn.Module):
    def __init__(self, embedding_dim) :
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            
            nn.ReLU(),
            
            nn.Linear(embedding_dim * 4, embedding_dim)
        )        
        
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, embedding_dim, num_head, block_size) :
        super().__init__()
        
        head_size = embedding_dim // num_head
        
        self.sa = MultiHead(num_head, head_size, embedding_dim, block_size)
        
        self.ffwd = FFN(embedding_dim)
        
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        self.ln2 = nn.LayerNorm(embedding_dim)
        
        
    def forward(self, x):
        x  = x +  self.sa(self.ln1(x))
        
        x = x + self.ffwd(self.ln2(x))
        
        return x
        
        
class LMHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        self.lm = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        output = self.lm(x)
        return output   


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, num_head, num_layer) :
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 
        
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        
        self.blocks = nn.ModuleList(
            [Block(embedding_dim, num_head, block_size) for _ in range(num_layer)]
        )
                
        self.lm = LMHead(embedding_dim, vocab_size)

        
    def forward(self, x):
        B, T = x.shape
        
        token = self.token_embedding(x)
        
        positions = torch.arange(T, device=x.device)
        
        pos = self.position_embedding(positions)
        
        output = token + pos
        
        for bloc in self.blocks:
            output = bloc(output)
                
        logits = self.lm(output)
        
        return logits
    
    def generate(self, x):  
        max_range = 100
        for i in range(max_range):
            x = x[:, -self.block_size:]
                    
            logits = self(x)
            
            last_logit = logits[:, -1, : ]
            
            probs = F.softmax(last_logit, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
                        
            x = torch.cat((x, next_token), dim=1) 
            if next_token.item() == stoi['<EOS>']:
                break
            
        return x
    
vocab = torch.load(f"{save_dir}/vocab.pt")
config = torch.load(f"{save_dir}/config.pt")

num_chunks = config["num_chunks"]

vocab_size = len(vocab)
embedding_dim = 64
block_size = 128
num_layer = 4
batch_size = 64
num_head = 4

model = GPT(vocab_size, embedding_dim, block_size, num_head, num_layer).to(device)
print("Model Creating..") 


optimiser = optim.Adam(
    model.parameters(),
    lr=3e-4 
)

total = sum(p.numel() for p in model.parameters())

print(f"Parameters : {total}")
print(f"{total/1e6:.2f} Million Parameters")

best_val_loss = float('inf')

for epoch in range(10):
    print(f"Epoch : {epoch + 1}")
    for chunk_id in range(num_chunks):
        encoded = torch.load(
            f"{save_dir}/chunk_{chunk_id}.pt"
        )

        n = int(0.9 * len(encoded))

        train_data = encoded[:n]
        val_data = encoded[n:]


        train_dataset = TextDataset(train_data, block_size)


        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=True
        )


        val_dataset = TextDataset(val_data, block_size)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
                    
        
        model.train()
        
        train_loss = 0
        
        for x, y in train_loader:
            
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            
            logits = logits.reshape(-1, vocab_size)
            
            target = y.reshape(-1).to(device)
            
            loss = F.cross_entropy(
                logits,
                target
            )
            
            train_loss += loss.item()
            
            optimiser.zero_grad()
            
            loss.backward()
            
            optimiser.step()
        
        train_loss /= len(train_loader)
            
        print(f"Train Loss : {train_loss:.4f}")
        
        
        model.eval()
        
        val_loss = 0
        
        with torch.no_grad():
            
            for x, y in val_loader:
                
                x = x.to(device)
                y = y.to(device)
                
                logits = model(x)
                
                logits = logits.reshape(-1, vocab_size)
                    
                y = y.reshape(-1)
                
                loss = F.cross_entropy(logits, y)
                
                val_loss += loss.item()

                
        val_loss /= len(val_loader)
        
        print(f"Val Loss : {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            print("Model Saved..")
            
            torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "embedding_dim": embedding_dim,
            "block_size": block_size,
            "num_layer": num_layer,
            "num_head": num_head,
        },
        f"{save_dir}/best_model.pth""best_model.pth"
        )
                
        print(f"{chunk_id} traning completed..")
                
                    
        del encoded
        del train_loader
        del val_loader
        del train_dataset
        del val_dataset
        del train_data
        del val_data
        
        torch.cuda.empty_cache()
    
print("All chunks Training completed..")