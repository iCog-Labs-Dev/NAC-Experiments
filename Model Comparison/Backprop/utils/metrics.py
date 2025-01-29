import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

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

            output = model(masked_data)

            if isinstance(output, tuple):
                reconstructed = output[0]  
            else:
                reconstructed = output

            reconstructed = reconstructed.view(data.size(0), -1)
            mse = F.mse_loss(reconstructed[~mask], data[~mask], reduction="sum")
            total_mse += mse.item() * data.size(0)
            total_samples += data.size(0)
            total_masked_elements += (~mask).sum().item()

    avg_mse = total_mse / (total_samples * data.size(1) // 2)

    return avg_mse

def extract_latents(encoder, dataloader):
    """
    Extracts latent representations from a trained encoder.
    
    Parameters:
    encoder: Trained encoder model.
    dataloader: Dataloader containing the dataset.

    Returns:
        Extracted latent representations, and corresponding labels.
    """
    encoder.eval()
    latents, labels = [], []

    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            batch_X = batch_X 
            batch_X = (batch_X > 0.5).float()  

            output = encoder(batch_X)

            if isinstance(output, tuple):  
                Z = output[0] 
            else:
                Z = output 

            latents.append(Z.view(Z.size(0), -1).cpu().numpy()) 
            labels.append(batch_Y.cpu().numpy())

    return np.vstack(latents), np.hstack(labels) 


def classification_error(encoder, train_loader, test_loader):
    """
    Computes the classification error using a log-linear model (logistic regression)
    fit to the latent representations.

    Parameters:
    encoder: Trained encoder model.
    train_loader: Training dataloader.
    test_loader: Testing dataloader.

    Returns:
    float: Classification error in percentage.
    """
    Z_train, Y_train = extract_latents(encoder, train_loader)
    Z_test, Y_test = extract_latents(encoder, test_loader)

    classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    classifier.fit(Z_train, Y_train)

    Y_pred = classifier.predict(Z_test)

    error = 1 - accuracy_score(Y_test, Y_pred)

    return error * 100  