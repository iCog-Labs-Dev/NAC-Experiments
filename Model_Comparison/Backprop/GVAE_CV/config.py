import torch

# Initialize model parameters
input_dim = 28 * 28
hidden_dim = [360, 360]
latent_dim = 20
l2_lambda = 1e-3
fixed_variance=torch.tensor(0.0)
learning_rate = 0.02
num_epochs = 50
n_components=75
num_samples = 5000
batch_size = 200