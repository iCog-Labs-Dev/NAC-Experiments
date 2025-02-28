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

def classification_error(model, train_loader, test_loader):
    model.eval()

    def extract_latents(loader, name=""):
        latent_representations = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                data, target = batch
                data = data.view(data.size(0), -1)

                # Convert one-hot encoded targets to class indices if needed
                if target.ndim > 1:  
                    target = torch.argmax(target, dim=1)
                else: 
                    target = target

                encoder_output = model.encoder(data)
                if isinstance(encoder_output, tuple):
                    mu = encoder_output[0] 
                else:
                    mu = encoder_output

                latent_representations.append(mu.cpu().numpy())
                labels.append(target.cpu().numpy())
        X = np.vstack(latent_representations)
        y = np.hstack(labels)
        # print(f"{model_name} - {name} latent shape: {X.shape}, label shape: {y.shape}")
        # print(f"{model_name} - {name} latent sample (first 5): {X[0, :5]}")
        # print(f"{model_name} - {name} label sample (first 5): {y[:5]}")
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

    print(f"Predicted labels (first 5): {Y_pred[:5]}")
    print(f"True labels (first 5): {Y_test[:5]}")
    return error_percentage
