import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from task_two import preprocess_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def log(msg):
    print(f"[INFO] {msg}", flush=True)



class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)



class LMDataset(Dataset):
    def __init__(self, texts, vocab, seq_len=20):
        self.data = []
        self.vocab = vocab

        for text in texts:
            tokens = ["<s>"] + text.split() + ["</s>"]
            ids = [vocab[t] for t in tokens if t in vocab]

            for i in range(len(ids) - seq_len):
                self.data.append((
                    torch.tensor(ids[i:i+seq_len]),
                    torch.tensor(ids[i+1:i+seq_len+1])
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_vocab(texts):
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
    for text in texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def train_model(model, dataloader, epochs=3, lr=0.001, log_every=200):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()

    log(f"Starting training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % log_every == 0:
                log(
                    f"Epoch {epoch} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        log(f"Epoch {epoch} completed | Avg loss: {avg_loss:.4f}")

    total_time = time.time() - start
    log(f"Training finished in {total_time:.2f} seconds")

    return total_time



def perplexity(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    total_tokens = 0

    log("Computing perplexity...")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()
            total_tokens += y.numel()

            if batch_idx % 500 == 0:
                log(f"Perplexity progress: batch {batch_idx}/{len(dataloader)}")

    ppl = math.exp(total_loss / total_tokens)
    log(f"Perplexity computation finished: {ppl:.2f}")

    return ppl



def generate_text(model, vocab, max_len=20):
    inv_vocab = {v: k for k, v in vocab.items()}
    model.eval()

    idx = vocab["<s>"]
    input_ids = torch.tensor([[idx]]).to(DEVICE)
    generated = []

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            probs = torch.softmax(logits[0, -1], dim=0)
            idx = torch.multinomial(probs, 1).item()

            if inv_vocab[idx] == "</s>":
                break

            generated.append(inv_vocab[idx])
            input_ids = torch.cat(
                [input_ids, torch.tensor([[idx]]).to(DEVICE)], dim=1
            )

    return " ".join(generated)


if __name__ == "__main__":
    log(f"Using device: {DEVICE}")

    df = preprocess_dataset("train.csv")
    texts = [t for t in df["clean_text"].tolist() if t.strip()]

    log(f"Loaded {len(texts)} documents")

    vocab = build_vocab(texts)
    log(f"Vocabulary size: {len(vocab)}")

    dataset = LMDataset(texts, vocab)
    log(f"Total training sequences: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    log(f"Batches per epoch: {len(loader)}")

    model = NeuralLanguageModel(len(vocab)).to(DEVICE)
    log("Model initialized")

    train_time = train_model(model, loader)
    ppl = perplexity(model, loader)

    log("Generating sample text:")
    print(generate_text(model, vocab))

    log(f"Training time: {train_time:.2f}s")
    log(f"Perplexity: {ppl:.2f}")

