# Backpropagation Models

## Overview
This directory contains the backpropagation based models we used for comparing generative and classification performances with the NGC models.
It focuses on the following autoencoder and variational autoencoder architectures:
- Variational Autoencoder (VAE)
- Generative Variational Autoencoder with Cross-Validation (GVAE-CV)
- Regularized Autoencoder (RAE)
- Generative Adversarial Network Autoencoder (GAN-AE)

### Training a Model
To train a model, use the following command:
```bash
python train_model.py --model=MODEL_TYPE --config=PATH_TO_CONFIG
```
### Evaluating a Model
To evaluate a trained model:
```bash
python eval_model.py --model=MODEL_TYPE --config=PATH_TO_CONFIG
```
## Evaluation Metrics
The following evaluation metrics were used to assess model performance:
 1. Binary Cross-Entropy (BCE): Used to evaluate the model's ability to reconstruct binary data
 2. Masked Mean Squared Error (M-MSE): Computes the Mean Squared Error on a partially masked input
 3. Classification Error (%Err): Measures the classification performance using the learned representations
 4. Log-Likelihood (Log-px): Provides insights into the model's ability to capture the underlying data distribution
