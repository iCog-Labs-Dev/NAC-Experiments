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


#another implementation of classification error without binarization
def classification_error(model, train_loader, test_loader, model_name="Unknown"):
    model.eval()

    def extract_latents(loader, name=""):
        latent_representations = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                data, target = batch
                data = data.view(data.size(0), -1)

                # Convert one-hot encoded targets to class indices if needed
                if target.ndim > 1:  # If one-hot encoded (e.g., shape [batch_size, 10])
                    target = torch.argmax(target, dim=1)
                else:  # Already class indices
                    target = target

                encoder_output = model.encoder(data)
                if isinstance(encoder_output, tuple):
                    mu = encoder_output[0]  # Take mu from (mu, logvar)
                else:
                    mu = encoder_output

                latent_representations.append(mu.cpu().numpy())
                labels.append(target.cpu().numpy())
        X = np.vstack(latent_representations)
        y = np.hstack(labels)
        print(f"{model_name} - {name} latent shape: {X.shape}, label shape: {y.shape}")
        print(f"{model_name} - {name} latent sample (first 5): {X[0, :5]}")
        print(f"{model_name} - {name} label sample (first 5): {y[:5]}")
        return X, y

    X_train, Y_train = extract_latents(train_loader, "Train")
    X_test, Y_test = extract_latents(test_loader, "Test")

    assert X_train.shape[0] == Y_train.shape[0], "Mismatch in training samples!"
    assert X_test.shape[0] == Y_test.shape[0], "Mismatch in test samples!"

    classifier = LogisticRegression(max_iter=1000, multi_class="multinomial")
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    error_percentage = 100 * (1 - accuracy)

    print(f"{model_name} - Predicted labels (first 5): {Y_pred[:5]}")
    print(f"{model_name} - True labels (first 5): {Y_test[:5]}")
    return error_percentage
