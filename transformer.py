import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model % 2 != 0:
            pe = torch.zeros(max_len, d_model+1)
        else:
            pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        print(div_term.shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if d_model % 2 != 0:
            pe = pe[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DroughtNetTransformer(nn.Module):

    def __init__(self, output_size, num_input_features, hidden_dim, n_layers, ffnn_layers,
        drop_prob, static_dim, num_heads, input_length, init_dim=128):
        super(DroughtNetTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.init_linear = nn.Linear(num_input_features, init_dim)
        self.pos_encoder = PositionalEncoding(init_dim, drop_prob, input_length)
        encoder_layers = TransformerEncoderLayer(init_dim, num_heads, hidden_dim, drop_prob)
        encoder_norm = LayerNorm(init_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers, encoder_norm)
        self.ninp = num_input_features
        # todo: add several layers given ffnn_layers
        # todo: add static
        self.ffnn_layers = []
        if ffnn_layers == 1:
            self.final = nn.Linear(init_dim*input_length, output_size)
        else:
            self.final = nn.Linear(hidden_dim, output_size)
            
        for i in range(ffnn_layers-1):
            if i == 0:
                self.ffnn_layers.append(nn.Linear(init_dim*input_length+static_dim, hidden_dim))
            else:
                self.ffnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
                
        self.ffnn_layers = nn.ModuleList(self.ffnn_layers)

        self.init_dim = init_dim
        self.input_length = input_length
        self.output_size = output_size
        
        self.init_weights()

    def init_weights(self):
        pass # possibly use initalization here

    def forward(self, x, static=None):
        # todo add static
        batch_size = x.size(0)
        x = x.cuda().to(dtype=torch.float32)
        if static is not None:
            static = static.cuda().to(dtype=torch.float32)
        x = self.init_linear(x)
        x = x * math.sqrt(self.ninp)
        output = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = output.reshape(
            batch_size,
            self.init_dim*self.input_length
        )
        for i in range(len(self.ffnn_layers)):
            if i == 0 and static is not None:
                output = self.ffnn_layers[i](torch.cat((output, static), 1))
            else:
                output = self.ffnn_layers[i](output)
        output = self.final(output)
        return output