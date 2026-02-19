import torch
import torch.nn as nn
from .embeddings import TokenEmbedding, PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    """Complete Transformer model from 'Attention is All You Need'"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        # Encoder and Decoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        # Share weights between embeddings and pre-softmax linear transformation
        self.tgt_embedding.embedding.weight = self.fc_out.weight
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()