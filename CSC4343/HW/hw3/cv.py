from sklearn.model_selection import KFold
import numpy as np


def cross_validate(data, epochs=10, n_splits=3, pretrain=True):
    # Unpack the data
    antigen_seqs, tcr_seqs, interactions = data

    # Convert lists to tensors for processing in PyTorch models
    antigen_tensor = torch.tensor(antigen_seqs, dtype=torch.long)
    tcr_tensor = torch.tensor(tcr_seqs, dtype=torch.long)
    interaction_tensor = torch.tensor(
        interactions, dtype=torch.float32).unsqueeze(1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(antigen_tensor):
        # Split data into training and test sets
        train_antigen, test_antigen = antigen_tensor[train_index], antigen_tensor[test_index]
        train_tcr, test_tcr = tcr_tensor[train_index], tcr_tensor[test_index]
        train_interaction, test_interaction = interaction_tensor[
            train_index], interaction_tensor[test_index]

        # Initialize models
        m_ant = make_tfm_model('antigen')
        m_tcr = make_tfm_model('tcr')

        # Optionally pretrain models
        if pretrain:
            pretrain_tfm_model(m_ant, train_antigen.tolist(), epochs)
            pretrain_tfm_model(m_tcr, train_tcr.tolist(), epochs)

        # Create and train the prediction model
        model = make_predict_model(m_ant, m_tcr)
        train_model(model, train_antigen.tolist(), train_tcr.tolist(),
                    train_interaction.squeeze(1).tolist(), epochs)

        # Evaluate the model
        preds = predict(model, test_antigen.tolist(), test_tcr.tolist())
        accuracy = (preds == test_interaction.squeeze(
            1).int().tolist()).float().mean()
        accuracies.append(accuracy.item())

    return np.mean(accuracies)


def main():
    path_to_data = './Data/data.csv'
    data = preprocess_data(path_to_data)

    # Cross-validate with pre-training
    accuracy_with_pretrain = cross_validate(data, pretrain=True)
    print(f'Accuracy with pre-training: {accuracy_with_pretrain}')

    # Cross-validate without pre-training
    accuracy_without_pretrain = cross_validate(data, pretrain=False)
    print(f'Accuracy without pre-training: {accuracy_without_pretrain}')


if __name__ == "__main__":
    main()
