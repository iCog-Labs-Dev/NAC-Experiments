# Backpropagation Models

## Overview
This directory contains backpropagation-based models used to compare generative and classification performance with Neural Generative Coding (NGC) models. The models implemented include:
- Gaussian Variational Autoencoder (GVAE)
- Gaussian Variational Autoencoder with Constant-Variance (GVAE-CV)
- Regularized Autoencoder (RAE)
- Generative Adversarial Network Autoencoder (GAN-AE)

## Training a Model
To train a model, use the following command:
```
python train_model.py --model=MODEL_TYPE --config=PATH_TO_CONFIG
```

Replace `MODEL_TYPE` with the specific model name (e.g., VAE, GVAE-CV, RAE, GAN-AE), and `PATH_TO_CONFIG` with the corresponding configuration file.

## Evaluating a Model
To evaluate a trained model:
```
python eval_model.py --model=MODEL_TYPE --config=PATH_TO_CONFIG
```
This will generate performance metrics based on the dataset and model configuration provided.

## Evaluation Metrics
The following evaluation metrics were used to assess model performance:
 1. **Binary Cross-Entropy (BCE)**: Used to evaluate the model's ability to reconstruct binary data
 2. **Masked Mean Squared Error (M-MSE)**: Computes the Mean Squared Error on a partially masked input
 3. **Classification Error (%Err)**: Measures the classification performance using the learned representations
 4. **Log-Likelihood (Log-px)**: Provides insights into the model's ability to capture the underlying data distribution
