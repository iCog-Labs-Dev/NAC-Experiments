# Neural Generative Coding (NGC) Experiments

## Overview

This research project aims to replicate and evaluate the Neural Generative Coding (NGC) framework proposed by Ororbia and Kifer, [The Neural Coding Framework for Learning Generative Models](https://arxiv.org/abs/2012.03405). The study provides a biologically inspired alternative to traditional backpropagation for training generative models.

## Background 

The NGC framework is rooted in predictive processing theory, offering a novel approach to machine learning that draws inspiration from biological neural systems. Unlike conventional training methods, NGC presents a unique perspective on how neural networks can learn and generate representations.

## Research Objectives 
- Replicating the NGC models described in the original study
- Evaluating model performance on benchmark datasets
- Comparing our results with the original study's findings
- Analyzing model performance across multiple evaluation metrics

## Key Metrics of Evaluation
1. **Reconstruction Quality**: Measures how well the model reconstructs input data
2. **Likelihood Estimation**: Evaluates the model's ability to generate realistic samples
3. **Classification Capability**: Assesses whether learned representations are useful for downstream classification tasks

## Installation
To set up the environment and run experiments, follow these steps:
1. Clone the Repository
```
git clone https://github.com/iCog-Labs-Dev/NAC-Experiments.git
cd NAC-Experiments
```
2. Build the Docker Image
```
docker build -t nac-experiments .
```
3. Run the Container
```
docker run --rm -it nac-experiments
```
## Citation
If you use this project in your research, please cite:

_**Ororbia, A. & Kifer, D. (2022). The neural coding framework for learning generative models.
Nature Communications, 13(1), 2064. DOI:**_ 10.1038/s41467-022-29632-7
<pre>
@article{Ororbia2022,
  author={Ororbia, Alexander and Kifer, Daniel},
  title={The neural coding framework for learning generative models},
  journal={Nature Communications},
  year={2022},
  month={Apr},
  day={19},
  volume={13},
  number={1},
  pages={2064},
  issn={2041-1723},
  doi={10.1038/s41467-022-29632-7},
  url={https://doi.org/10.1038/s41467-022-29632-7}
}
</pre>

# License
ngc-learn is distributed under the BSD 3-Clause License.
