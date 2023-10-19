import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout_rate=0.1):
        super(ConformerBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(d_model, 2 * d_model, kernel_size=3, padding=1, bias=False)
        self.glu = nn.GLU(dim=1)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # multi-head self-attention
        attn_output, _ = self.multihead_attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # feedforward network
        feedforward_output = self.feedforward(x)
        x = x + feedforward_output
        x = self.layer_norm2(x)

        # convolutinal block
        residual = x
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.glu(x)
        x = x.transpose(1, 2)
        x = self.linear2(F.relu(self.linear1(x)))
        x = self.layer_norm3(x + residual)
        x = self.dropout(x)

        return x

class Conformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_head, dim_feedforward, num_layers, dropout_rate=0.1):
        super(Conformer, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1, bias=False)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout_rate)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(d_model, n_head, dim_feedforward, dropout_rate) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # convolutional layer
        x = self.conv1d(x)

        # positional encoding
        x = x + self.pos_embedding[:, :, :x.size(2)]

        # conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        # global average pooling
        x = x.mean(dim=2)

        # linear layer
        x = self.fc(x)

        return x