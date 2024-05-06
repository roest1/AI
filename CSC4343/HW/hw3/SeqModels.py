'''
Results from 3-fold cross-validation:
-------------------------------------
                       With Pre-training         Without Pre-training

Fold 1 Accuracy = 
Fold 2 Accuracy = 
Fold 3 Accuracy = 

Average Accuracy = 

(Discuss whether performance gain by pre-training meets your expectations)

The training of one model took my computer almost two whole days and I wasn't able to do the cross validation.

Also, I'm not sure why, but the model isn't being correctly read from my google drive even though I set the permissions.
'''

import torch.nn as nn
import torch
import requests
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def encode_sequence(seq, vocab):
    return [vocab[char] for char in seq]


def preprocess_data(path_to_data:str):
    '''
    preprocess_data()

    Arguments:
    - path_to_data: Path to the data file that contains the TCR-antigen interaction data

    Returns:
    - Preprocessed data with sequences padded with "X" so that all TCR sequences
    are the same length and all antigen sequences have the same length
    '''
    data = pd.read_csv(path_to_data)
    max_len_antigen = data['antigen'].apply(len).max()
    max_len_tcr = data['TCR'].apply(len).max()

    # Define vocabulary for amino acids plus padding
    vocab = {aa: idx + 1 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 1-indexed
    vocab['X'] = 0  # Padding index

    antigen_seqs = [encode_sequence(
        seq + 'X' * (max_len_antigen - len(seq)), vocab) for seq in data['antigen']]

    tcr_seqs = [encode_sequence(
        seq + 'X' * (max_len_tcr - len(seq)), vocab) for seq in data['TCR']]

    interactions = data['interaction'].tolist()

    return antigen_seqs, tcr_seqs, interactions


def make_tfm_model(T:str) -> nn.Module:
    '''
    make_tfm_model()

    Arguments:
    - T: string with value either "tcr" or "antigen"

    Returns:
    - PyTorch object that implements the transformer model for the corresponding
    tcr/antigen sequences
    '''
    # Use nn.TransformerEncoder with 3 layers of 
    # nn.TransformerEncoderLayer.
    # Each layer has d_model=256 and nhead=8
    # Use nn.Embedding layer to translate the AA tokens to embedding vectors.
    
    class SequentialTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(21, 256) # 20 AA's + 1 padding
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=256, nhead=8, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=3)

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer_encoder(x)
            return x
    
    return SequentialTransformer() # Model is the same for both tcr and antigen cases
        

def make_predict_model(M_antigen:nn.Module, M_tcr:nn.Module) -> nn.Module:
    '''
    make_predict_model()

    Arguments:
    - M_antigen: Antigen transformer model
    - M_tcr: TCR transformer model

    Returns:
    - Full model that makes the prediction on the TCR-antigen interaction
    '''
    # M_antigen and M_tcr both get appended a 1d tensor
    # of size 256 (sequential) from the last position. 
    # Then the two models get concatenated
    # After the concatenation, there is 1 linear layer of 256 neurons + 1 output layer
    
    class InteractionModel(nn.Module):
        def __init__(self, M_antigen, M_tcr):
            super().__init__()
            self.antigen_model = M_antigen
            self.tcr_model = M_tcr
            self.classifier = nn.Sequential(
                nn.Linear(512, 256), 
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        def forward(self, antigen_seq, tcr_seq):
            antigen_seq = self.antigen_model(antigen_seq)[:, -1, :] # take final state
            tcr_seq = self.tcr_model(tcr_seq)[:, -1, :] # take final state
            x = torch.cat((antigen_seq, tcr_seq), dim=1)
            x = self.classifier(x)
            return x
    
    return InteractionModel(M_antigen, M_tcr)



def pretrain_tfm_model(M:nn.Module, D:list, n_epochs:int):
    '''
    pretrain_tfm_model()

    Arguments:
    - M: Transformer model
    - D: Training Dataset
    - n_epochs: Number of epochs to pretrain the model for

    This function prints out the average loss during each epoch of pre-training
    '''
    optimizer = torch.optim.Adam(M.parameters())
    criterion = nn.CrossEntropyLoss()

    M.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for sequence in D:
            inputs = torch.tensor([sequence[:-1]], dtype=torch.long)
            targets = torch.tensor([sequence[1:]], dtype=torch.long)
            outputs = M(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Avg Loss = {total_loss / len(D)}')

def train_model(M:nn.Module, L_antigen:list, L_tcr:list, Interaction:list, n_epochs:int):
    '''
    train_model()

    Arguments:
    - M: Full model that makes the prediction on the TCR-antigen interaction
    - L_antigen: A list of antigen sequences
    - L_tcr: A list of TCR sequences
    - Interaction: A list of numbers (0/1) indicating interaction or not
    - n_epochs: Number of epochs to train the model for

    This function prints out the average loss during each epoch of training
    '''
    dataset = TensorDataset(torch.tensor(L_antigen, dtype=torch.long), torch.tensor(L_tcr, dtype=torch.long), torch.tensor(Interaction, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(M.parameters())
    criterion = nn.BCELoss()

    M.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for antigen, tcr, interact in loader:
            outputs = M(antigen, tcr)
            loss = criterion(outputs.squeeze(), interact.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Avg Loss = {total_loss / len(L_antigen)}')

def load_trained_model_local(filepath:str) -> nn.Module:
    '''
    load_trained_model_local()

    Loads a trained model from the filepath
    '''
    m_tcr = make_tfm_model('tcr')
    m_ant = make_tfm_model('antigen')
    model = make_predict_model(m_ant, m_tcr)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_file_from_google_drive(file_id:str, destination:str):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def load_trained_model():
    '''
    load_trained_model()

    Downloads the trained model from google drive and loads it into memory
    '''
    file_id = '1xwXCZD-AF3cMVuL8oua2524bsJGeg2Lt'
    model_path = 'trained_model.pt'
    download_file_from_google_drive(file_id, model_path)
    return load_trained_model_local(model_path)


def save_model(model:nn.Module, filepath:str):
    '''
    save_model()

    Saves the model to the filepath locally
    '''
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def predict(M:nn.Module, L_antigen:list, L_tcr:list) -> list:
    '''
    predict()

    Arguments:
    - M: Full prediction model
    - L_antigen: List of antigen sequences
    - L_tcr: List of TCR sequences

    Returns:
    - A list of (0/1) prediction results 

    For each corresponding pair of sequences in L_antigen and L_tcr, this function
    uses the model to make a prediction on their interaction. 
    '''
    M.eval() 
    predictions = []
    with torch.no_grad():
        for antigen, tcr in zip(L_antigen, L_tcr):
            output = M(antigen, tcr)
            predicted = (output > 0.5).int()  # Convert probabilities to 0/1
            predictions.append(predicted)
    return predictions


def main():

    path_to_data = './Data/data.csv'
    list_antigen_seq, list_tcr_seq, list_interact = preprocess_data(path_to_data)
    
    # m_tcr = make_tfm_model('tcr')
    # m_ant = make_tfm_model('antigen')

    # pretrain_tfm_model(m_tcr, list_tcr_seq, 10)
    # pretrain_tfm_model(m_ant, list_antigen_seq, 10)


    # model = make_predict_model(m_ant, m_tcr)

    # train_model(model, list_antigen_seq, list_tcr_seq, list_interact, 10)
    # save_model(model, 'trained_model.pt')

    tm = load_trained_model()

    pred = predict(tm, list_antigen_seq, list_tcr_seq)
    print(f"Prediction = {pred}")

if __name__ == "__main__":
    main()