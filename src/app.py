import os
import pickle
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from collections import Counter

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
    def fit(self, texts):
        pass 
                
    def encode(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in text.split()]
            sequences.append(seq)
        return sequences

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

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pickle')

@st.cache_resource
def load_model_and_tokenizer(model_type='lstm'):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'model_{model_type}.pt')
    if not os.path.exists(model_path) or not os.path.exists(TOKENIZER_PATH):
        return None, None, None, None
    
    import __main__
    __main__.SimpleTokenizer = SimpleTokenizer
    
    with open(TOKENIZER_PATH, 'rb') as f:
        data = pickle.load(f)
        
    tokenizer = data['tokenizer']
    max_sequence_len = data['max_sequence_len']
    vocab_size = data['vocab_size']
    
    if model_type == 'lstm':
        model = LSTMModel(vocab_size)
    else:
        model = RNNModel(vocab_size)
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, tokenizer, max_sequence_len, vocab_size


st.set_page_config(page_title="Next Word Predictor", layout="centered", page_icon="📝")

st.markdown("""
<style>
    .title-box { text-align: center; font-family: 'Inter', sans-serif; color: #2e3b4e; }
    .prediction-highlight { 
        color: #e04f5f; font-weight: bold; background-color: #fee1e8; 
        padding: 5px 10px; border-radius: 8px; font-size: 1.3em;
    }
    .stButton>button { width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-box'>📝 AI Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Switch between RNN and LSTM to compare language models.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    selected_model_str = st.selectbox("🤖 Select Model Architecture:", ["LSTM (Long Short-Term Memory)", "RNN (Recurrent Neural Network)"])
    model_type = "lstm" if "LSTM" in selected_model_str else "rnn"

with col2:
    st.markdown("<br>", unsafe_allow_html=True) 

model, tokenizer, max_sequence_len, vocab_size = load_model_and_tokenizer(model_type)

user_input = st.text_input("Enter your text prefix:", placeholder="E.g., To be or not to")
predict_btn = st.button("🔮 Predict Next Word")

if predict_btn:
    if model is None:
        st.error(f"Cannot find the trained {model_type.upper()} model! Please wait until `train_models.py` finishes running or run it manually.")
    elif not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        # Preprocess input (tokenize, pad)
        text = user_input.lower()
        for p in ['.', ',', '!', ':', ';']:
            text = text.replace(p, f' {p} ')
            
        token_list = tokenizer.encode([text])[0]
        
        # Pad sequence
        seq_length = max_sequence_len - 1
        if len(token_list) < seq_length:
            token_list = [0] * (seq_length - len(token_list)) + token_list
        else:
            token_list = token_list[-seq_length:]
            
        token_tensor = torch.tensor([token_list], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(token_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
            
        top_k_indices = np.argsort(probs)[-5:][::-1]
        
        predictions = []
        for index in top_k_indices:
            idx = int(index)
            if tokenizer.idx2word[idx] not in ["<PAD>", "<UNK>"]:
                predictions.append((tokenizer.idx2word[idx], probs[idx]))
                
        st.write("### 🎯 Predicted Next Word Options:")
        
        if len(predictions) > 0:
            top_word = predictions[0][0]
            st.markdown(f"<div style='text-align: center; margin: 20px 0;'><span class='prediction-highlight'>{top_word}</span></div>", unsafe_allow_html=True)
            
            st.write("#### 📊 Alternate Options (Top 5 Probabilities):")
            
            cols = st.columns(len(predictions))
            for i, (word, prob) in enumerate(predictions):
                with cols[i]:
                    st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);'>"
                                f"<strong style='color: #1f77b4; font-size: 1.1em;'>{word}</strong><br>"
                                f"<span style='color: #888; font-size: 0.9em;'>{prob*100:.1f}%</span>"
                                f"</div>", unsafe_allow_html=True)
        else:
            st.info("Model could not confidently predict the next word.")
