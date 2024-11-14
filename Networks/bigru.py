import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
BiLSTM网络
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
https://github.com/xashru/mixup-text/blob/master/models/text_lstm.py
"""

hidden_layers = 2


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=2, dropout=0.5, bid=True):
        super(BiGRU, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_hidden = embedding_dim

        # GRU layer
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=hidden_layers,
                          dropout=dropout,
                          bidirectional=bid)
        self.fc1 = nn.Linear(self.n_hidden * hidden_layers * (1 + bid), embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

        self.optimizer = optim.Adam(self.parameters(), 0.001)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, text, perturbation=None):
        emb = self.embeddings(text)
        if perturbation is not None and perturbation != 'advtext':
            emb = self.dropout(emb)
            emb += perturbation
        x = text.permute(1, 0, 2)

        _, x = self.gru(x)

        pred = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        pred = self.fc1(pred)
        out = self.fc2(pred)

        if perturbation is not None:
            return emb, out
        else:
            return pred, out