import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluate_perc_err(model, train_loader, test_loader):
    """
    Evaluate a model's classification performance using latent representations and logistic regression.

    This function extracts latent representations from the encoder of the given model for both training and testing datasets.
    It then trains a logistic regression classifier on the latent representations of the training data and evaluates its 
    classification error on the test data.

    Args:
        model: The trained model containing an encoder to extract latent representations.
        train_loader: DataLoader providing the training dataset, with each batch containing data and corresponding labels.
        test_loader: DataLoader providing the testing dataset, with each batch containing data and corresponding labels.

    Returns:
        err: Classification error (percentage) on the test dataset.
        
    """
    model.eval()
    train_latents, train_labels = [], []
    test_latents, test_labels = [], []

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