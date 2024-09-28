import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, max_seq_length=128):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)

        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # src is expected to have shape (batch_size, seq_len)
        seq_len = src.size(1)

        # Get embeddings and add positional encoding
        src = self.embedding(src) + self.positional_encoding[:, :seq_len, :]  # Add positional encoding
        output = self.transformer_encoder(src)

        output = self.fc_out(output)
        return output

    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding for input embeddings"""
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)  # Shape: (1, max_len, d_model)

