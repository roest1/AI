'''
CSC4343 HW3 Reference Implementation

This reference implementation uses a small number of epochs (10) to test the code.
To fully explore the model capability, in your homework, you should try a larger number of epochs
for both the pretraining and the training of the full model.
'''

import numpy as np, pandas as pd
from dataclasses import dataclass
@dataclass
class Config:
    d_model: int
    nhead: int
    d_hid: int
    nlayers: int
    d_pred: int
    cuda: str
    batch_size: int
    lr: float

cfg = Config(d_model = 256,
             nhead = 8,
             d_hid = 512,
             nlayers = 3,
             d_pred = 128,
             cuda = 'cuda',
             batch_size = 128,
             lr = 1e-4
            )


aa_to_int = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 0
}
Maxlen = {'antigen': 12,
          'tcr': 20}

def convert_seq(sequences, T):
    cc = []
    for s in sequences:
        e = [aa_to_int.get(aa, 0) for aa in s]
        padded = e + [aa_to_int['X']] * (Maxlen[T] - len(e))
        cc.append(torch.tensor(padded))
    return torch.stack(cc).to(cfg.cuda)

def read_data(return_tensor=False):
    antigens, tcrs, labels = [], [], []
    data = pd.read_csv('data.csv')
    for k, d in data.iterrows():
        antigens.append(d['antigen'])
        tcrs.append(d['TCR'])
        labels.append(d['interaction'])

    if return_tensor:
        return convert_seq(antigens, 'antigen'), convert_seq(tcrs, 'tcr'), torch.tensor(labels).to(cfg.cuda)
    else:
        return antigens, tcrs, labels

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

def make_dataloader(*data, shuffle=True):
    dataset = TensorDataset(*data)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)

def train(m, opt, dl, loss_func, nepochs):
    m.train()
    for i in range(nepochs):
        epoch_loss = 0
        for b in dl:
            loss = loss_func(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f'Epoch {i}, loss = {epoch_loss/len(dl):.3f}')


class PositionalEncoding(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, cfg.d_model, 2) * (-np.log(10000.0) / cfg.d_model))
        pe = torch.zeros(max_len, 1, cfg.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: Tensor, shape [seq_len, batch_size, embedding_dim]
        return x + self.pe[:x.size(0)]


class TransformerModel(nn.Module):
    def __init__(self, seq_type):
        super().__init__()
        self.seq_type = seq_type
        self.embedding = nn.Embedding(len(aa_to_int), cfg.d_model)
        self.pos_encoder = PositionalEncoding(Maxlen[seq_type])
        encoder_layers = nn.TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.d_hid, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layers, cfg.nlayers)

        self.pretrain_predict = nn.Linear(cfg.d_model, len(aa_to_int))


    def forward(self, src):
        # First convert input tensor src from [batch_size, seq_len] to [seq_len, batch_size]
        src = self.embedding(src.transpose(0, 1)) * np.sqrt(cfg.d_model)
        src = self.pos_encoder(src)

        # Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        # Unmasked positions are filled with float(0.0).
        src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        output = self.transformer(src, src_mask)

        # Convert back to [batch_size, seq_len, embedding] before return
        output = output.transpose(0, 1)
        return output

    def pretrain_pred(self, src):
        return self.pretrain_predict(self.forward(src))


def make_tfm_model(T):
    return TransformerModel(T).to(cfg.cuda)

def pretrain_tfm_model(M, D, n_epochs):
    seq = convert_seq(D, M.seq_type)
    dl = make_dataloader(seq)
    xentloss = nn.CrossEntropyLoss()
    opt = Adam(M.parameters(), lr=cfg.lr)
    def loss_func(b):
        b = b[0]
        pretrain_pred = M.pretrain_pred(b[:, :-1]).transpose(1, 2)
        return xentloss(pretrain_pred, b[:, 1:])

    train(M, opt, dl, loss_func, n_epochs)


class AntigenTCRModel(nn.Module):
    def __init__(self, M_antigen, M_tcr):
        super().__init__()
        self.m_antigen = M_antigen
        self.m_tcr = M_tcr

        self.predict = nn.Sequential(
            nn.Linear(2*cfg.d_model, cfg.d_pred),
            nn.ReLU(),
            nn.Linear(cfg.d_pred, 2)
        )

    def forward(self, antigen, tcr):
        t_antigen = self.m_antigen(antigen)[:, -1, :]
        t_tcr = self.m_tcr(tcr)[:, -1, :]

        t = torch.cat([t_antigen, t_tcr], dim=1)
        return self.predict(t)

    def predict_label(self, antigen, tcr, T=0.5):
        pred = self.forward(antigen, tcr)[:, 1]
        return (pred >= 0.5).int().cpu().numpy()


def make_predict_model(M_antigen, M_tcr):
    return AntigenTCRModel(M_antigen, M_tcr).to(cfg.cuda)

def train_model(M, L_antigen, L_tcr, Interaction, n_epochs):
    antigen = convert_seq(L_antigen, 'antigen')
    tcr = convert_seq(L_tcr, 'tcr')
    label = torch.tensor(Interaction).to(cfg.cuda)
    dl = make_dataloader(antigen, tcr, label)
    xentloss = nn.CrossEntropyLoss()
    opt = Adam(M.parameters(), lr=cfg.lr)
    def loss_func(b):
        antigen, tcr, label = b
        pred = M(antigen, tcr)
        return xentloss(pred, label)

    train(M, opt, dl, loss_func, n_epochs)

def predict(M, L_antigen, L_tcr):
    antigen = convert_seq(L_antigen, 'antigen')
    tcr = convert_seq(L_tcr, 'tcr')
    dl = make_dataloader(antigen, tcr, shuffle=False)
    res = []
    M.eval()
    for a, t in dl:
        with torch.no_grad():
            res.extend(M.predict_label(a, t).tolist())
    return res


if __name__ == "__main__":

    a, t, l = read_data()

    ma = make_tfm_model('antigen')
    mt = make_tfm_model('tcr')

    pretrain_tfm_model(ma, a, 10)
    pretrain_tfm_model(mt, t, 10)

    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

    def split(d, train_idx, test_idx):
        return [d[i] for i in train_idx], [d[i] for i in test_idx]

    cv = KFold(3, shuffle=True)
    accs = 0
    for i, (train_index, test_index) in enumerate(cv.split(a)):
        antigen_train, antigen_test = split(a, train_index, test_index)
        tcr_train, tcr_test = split(t, train_index, test_index)
        y_train, y_test = split(l, train_index, test_index)

        ma_clone = make_tfm_model('antigen').load_state_dict(ma.state_dict())
        mt_clone = make_tfm_model('tcr').load_state_dict(mt.state_dict())
        model = make_predict_model(ma_clone, mt_clone)
        
        train_model(model, antigen_train, tcr_train, y_train, 10)
        y_pred = predict(model, antigen_test, tcr_test)
        acc = accuracy_score(y_test, y_pred)
        print(f'CV {i}, acc = {acc:.3f}')
        accs += acc

    print(f'mean CV acc = {accs/3:.3f}')
