
import torch
import torch.nn as nn
import torch.optim as optim

"""
RNN网络
Ref: https://github.com/shawroad/NLP_pytorch_project/tree/master/Text_Classification
https://github.com/xashru/mixup-text/blob/master/models/text_lstm.py
https://github.com/Lizhen0628/text_classification/blob/master/model/models.py
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_layers = 2


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes=2, dropout=0.5, bid=True):
        super(RNN, self).__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.n_hidden = embedding_dim

        # RNN layer
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=hidden_layers, dropout=dropout, batch_first=True,
                            bidirectional=bid)
        self.fc = nn.Linear(self.n_hidden * hidden_layers * (1 + bid), num_classes)
        
        self.optimizer = optim.Adam(self.parameters(), 0.001)
        # nn.CrossEntropyLoss会自动加上Sofrmax层
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, text, perturbation=None):
        # input x shape is [batch, seq_len]
        emb = self.embeddings(text)
        if perturbation is not None and perturbation != 'advtext':
            emb = self.dropout(emb)
            emb += perturbation

        output, x = self.rnn(emb)
        x = torch.cat([x[i, :, :] for i in range(x.shape[0])], dim=1)
        out = self.fc(x)

        if perturbation is not None:
            return emb, out
        else:
            return x, out