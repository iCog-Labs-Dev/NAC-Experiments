import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

def evaluate_perc_err(model, train_loader, test_loader):
    """
    Evaluate a model's classification performance using latent representations and logistic regression.

    This function extracts latent representations from the encoder of the given model for both training and testing datasets.
    It then trains a logistic regression classifier on the latent representations of the training data and evaluates its 
    classification error on the test data.

    Args:
        model: The trained model containing an encoder to extract latent representations.
        train_loader: DataLoader providing the training dataset.
        test_loader: DataLoader providing the testing dataset.

    Returns:
        err: Classification error (percentage) on the test dataset.
        
    """
    model.eval()
    train_latents, train_labels = [], []
    test_latents, test_labels = [], []
    
    # Extract latent representations from data
    with torch.no_grad():
        for data, label in train_loader:
            data = data.view(data.size(0), -1)
            if label is not None:
                z = model.encoder(data)
                train_latents.append(z.cpu().numpy())
                train_labels.append(label.cpu().numpy())
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.view(data.size(0), -1)
            if label is not None:
                z = model.encoder(data)
                test_latents.append(z.cpu().numpy())
                test_labels.append(label.cpu().numpy())

    train_latents = np.vstack(train_latents)
    train_labels = np.hstack(train_labels).reshape(-1)
    test_latents = np.vstack(test_latents)
    test_labels = np.hstack(test_labels).reshape(-1)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_latents, train_labels)

    predictions = clf.predict(test_latents)
    err = 100 * (1 - accuracy_score(test_labels, predictions))
    return err

def masked_mse(model, loader):
    """
    Computes the average masked mean squared error (MSE) loss for a model. 

    The first half of each input sample is masked, and the MSE is calculated 
    on the masked elements between the original and reconstructed data.

    Args:
        model: The model to evaluate.
        loader: DataLoader providing the input data.

    Returns:
        float: Average masked MSE loss across all samples.
    """

    model.eval()
    total_mse = 0.0
    total_samples = 0
    total_masked_elements = 0
    with torch.no_grad():
        for data, _ in loader:

            data = data.view(data.size(0), -1)
            data = (data > 0.5).float()
            mask = torch.ones_like(data, dtype=torch.bool)
            mask[:, : data.size(1) // 2] = 0

            masked_data = data * mask.float()
            masked_data = (masked_data > 0.5).float()
            reconstructed = model(masked_data)
            reconstructed = reconstructed.view(data.size(0), -1)

            mse = F.mse_loss(reconstructed[~mask], data[~mask], reduction="sum")
            total_mse += mse.item() * data.size(0)
            total_samples += data.size(0)
            total_masked_elements += (~mask).sum().item()

    avg_mse = total_mse / (total_samples * data.size(1) // 2)

    return avg_mse