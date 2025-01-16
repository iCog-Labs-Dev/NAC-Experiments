import torch
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle

class GMM:
    def __init__(self, k, max_iter=50, assume_diag_cov=False, init_kmeans=True):
        self.k = k
        self.max_iter = max_iter
        self.assume_diag_cov = assume_diag_cov
        self.init_kmeans = init_kmeans
        self.weights = None
        self.means = None
        self.covariances = None
        self.chol_decomps = None

    def fit(self, data):
        """
        Fits the GMM parameters using sklearn for initialization and implement EM optimization.
        Args:
            data (torch.Tensor): The input data (n_samples, n_features).
        """
        data_np = data.cpu().numpy()
        cov_type = 'diag' if self.assume_diag_cov else 'full'

        # Sklearn GMM for initialization
        gmm = GaussianMixture(n_components=self.k, max_iter=self.max_iter, covariance_type=cov_type, init_params='kmeans')
        gmm.fit(data_np)

        self.weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=data.device)
        self.means = torch.tensor(gmm.means_, dtype=torch.float32, device=data.device)
        self.covariances = torch.tensor(gmm.covariances_, dtype=torch.float32, device=data.device)
        self.chol_decomps = torch.linalg.cholesky(self.covariances)

        # EM optimization
        n_samples, n_features = data.size()
        for _ in range(self.max_iter):
            log_resp = self._estimate_log_prob(data) + torch.log(self.weights)
            log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
            resp = torch.exp(log_resp)

            nk = resp.sum(dim=0)
            self.weights = nk / n_samples
            self.means = (resp.t() @ data) / nk.unsqueeze(1)

            for k in range(self.k):
                diff = data - self.means[k]
                weighted_cov = (resp[:, k].unsqueeze(1) * diff).t() @ diff
                self.covariances[k] = weighted_cov / nk[k] + 1e-6 * torch.eye(n_features, device=data.device)
                self.chol_decomps[k] = torch.linalg.cholesky(self.covariances[k])