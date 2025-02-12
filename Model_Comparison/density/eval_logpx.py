import torch
import numpy as np


def evaluate_logpx(data_loader, model, gmm, latent_dim, num_samples=5000, batch_size=200, binarize_x=True):
    """
    Evaluate the marginal log-likelihood log p(x) for a dataset using a trained GMM and model.

    Args:
        dataloader: DataLoader providing the dataset.
        model: The trained model with an encoder and decoder.
        gmm: The trained GMM.
        latent_dim: Dimension of the latent space.
        num_samples: Number of Monte Carlo samples for estimating log p(x).
        batch_size: Batch size for processing the dataset.
        binarize_x: Whether to binarize the input data (True for binary cross-entropy).

    Returns:
        mean_logpx: The mean marginal log-likelihood log p(x) over the dataset.
    """
    logpx_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].to(device)
            data = (data > 0.5).float()
            for x in data:
                x = x.unsqueeze(0).to(device)
                z_samples = gmm.sample(num_samples).to(device)

                x_recon = model.decoder(z_samples)
                x_recon = torch.clamp(x_recon, 1e-6, 1 - 1e-6)

                x = x.view(x.size(0), -1)
                x_recon = x_recon.view(x_recon.size(0), -1)

                # Compute log p(x|z) using Binary Cross Entropy
                if binarize_x:
                    logp_xz = torch.sum(x * torch.log(x_recon) + (1 - x) * torch.log(1 - x_recon), dim=1)
                else:
                    sigma = 1.0
                    diff = (x - x_recon)
                    logp_xz = torch.sum(-(diff ** 2) / (2 * sigma**2) - 0.5 * torch.log(2 * np.pi * sigma**2), dim=1)

                logpx = torch.logsumexp(logp_xz, dim=0) - np.log(num_samples)
                logpx_list.append(logpx.item())

    mean_logpx = np.mean(logpx_list)
    return mean_logpx