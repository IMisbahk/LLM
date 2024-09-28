import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, max_seq_length=128):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, d_model))

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        seq_len = src.size(1)  # Number of tokens in the input sequence
        batch_size = src.size(0)  # Number of sequences in the batch
        
        # Ensure the positional encoding has the correct shape
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        src = self.embedding(src)  # Embed the input tokens
        src += pos_enc  # Add positional encoding

        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output
