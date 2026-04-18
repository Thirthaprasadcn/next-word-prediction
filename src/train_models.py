import os
import requests
import numpy as np
import pickle
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Constants
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.txt')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pt')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pickle')
SAMPLE_TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEQ_LENGTH = 7  # number of words to look back
EPOCHS = 30 # Increased for better accuracy
BATCH_SIZE = 128

def download_data_if_missing():
    """Download sample text if data file is empty or missing."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print("Downloading/Refreshing sample data (Tiny Shakespeare)...")
    response = requests.get(SAMPLE_TEXT_URL)
    response.raise_for_status()
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        # Increase text slice for better accuracy
        text_chunk = response.text[:200000]
        f.write(text_chunk)
    print("Download complete.")

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
    def fit(self, texts):
        words = []
        for text in texts:
            words.extend(text.split())
        word_counts = Counter(words)
        for word, _ in word_counts.items():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def encode(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in text.split()]
            sequences.append(seq)
        return sequences

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data():
    """Read data, tokenize and create sequences."""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        
    # Remove some punctuation for simplicity
    for p in ['.', ',', '!', ':', ';']:
        text = text.replace(p, f' {p} ')
        
    corpus = text.split('\n')
    corpus = [line.strip() for line in corpus if line.strip()]
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit(corpus)
    
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.encode([line])[0]
        # create n-grams
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[max(0, i - SEQ_LENGTH):i+1]
            input_sequences.append(n_gram_sequence)
            
    # Pad sequences
    max_sequence_len = min(max([len(x) for x in input_sequences]), SEQ_LENGTH + 1)
    
    padded_sequences = []
    for seq in input_sequences:
        if len(seq) < max_sequence_len:
            seq = [0] * (max_sequence_len - len(seq)) + seq
        else:
            seq = seq[-max_sequence_len:]
        padded_sequences.append(seq)
        
    padded_sequences = np.array(padded_sequences)
    
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]
    
    return X, y, max_sequence_len, tokenizer

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits

def main():
    print("Preparing data...")
    download_data_if_missing()
    
    X, y, max_sequence_len, tokenizer = preprocess_data()
    print(f"Total vocabulary size: {tokenizer.vocab_size}")
    print(f"Total sequences/samples: {len(X)}")
    
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Save tokenizer once
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump({
            'tokenizer': tokenizer, 
            'max_sequence_len': max_sequence_len,
            'vocab_size': tokenizer.vocab_size
        }, f)
        
    model_types = ['lstm', 'rnn']
    
    for mtype in model_types:
        print(f"\\n--- Building and Training {mtype.upper()} model on {device} ---")
        if mtype == 'lstm':
            model = LSTMModel(tokenizer.vocab_size).to(device)
        else:
            model = RNNModel(tokenizer.vocab_size).to(device)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
        
        save_path = MODEL_PATH.replace('model.pt', f'model_{mtype}.pt')
        print(f"Saving {mtype.upper()} model to {save_path}...")
        torch.save(model.state_dict(), save_path)
        
    print("\\nAll Training complete!")

if __name__ == "__main__":
    main()
