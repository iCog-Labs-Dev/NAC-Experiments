# NGC-Learn
NGC-Learn is a Python library for building, simulating, and analyzing predictive processing models using the Neural Generative Coding (NGC) framework. This toolkit is built on TensorFlow 2 and is designed for research in neurobiologically inspired learning.

## Features
- Supports predictive coding and other biologically motivated learning paradigms.
- Built on TensorFlow 2 for GPU-accelerated computation.
- Actively maintained by the <a href="https://www.cs.rit.edu/~ago/nac_lab.html">Neural Adaptive Computing (NAC) laboratory</a>.

## <b>Documentation</b>
- Official Docs: <a href="https://ngc-learn.readthedocs.io/en/stable/">here</a>
- Related Blog Post: <a href="https://go.nature.com/3rgl1K8">here</a>
- Source Paper: <a href="https://www.nature.com/articles/s41467-022-29632-7"> here</a>

## <b>Installation:</b>
### Standard Installation
Setup: Ensure that you have installed the following base dependencies in your system. Note that this library was developed on **Ubuntu 18.04** and tested on **Ubuntu(s) 16.04** and **18.04** (and should also work on Ubuntu 20.04). Ensure you have the following dependencies installed:

1. Python (>=3.7)
2. TensorFlow (>=2.0.0, GPU version recommended)
3. NumPy (>=1.20.0)
4. scikit-learn (>=0.24.2)
5. matplotlib (>=3.4.3)
6. networkx (>=2.6.3) (optional, for visualization)
7. pyviz (>=0.2.0) (optional, for visualization)

To install all dependencies, run: 
```
pip install -r requirements.txt
```
Then, install ngc-learn:
```
python setup.py install
```
### Using Docker
Alternatively, you can set up a pre-configured environment using Docker:
1. Navigate to the directory where the Dockerfile is located.
   ```
   cd NAC-Experiments/Model_Comparison/Stable_Version
   ```
2. Build the Docker Image
   
   ```
   docker build -t nac-experiments .
   ```
3. Run the Container
   
   ```
    docker run --rm -it nac-experiments
   ```
## <b>Usage</b>
To verify installation, run:
```
import ngclearn
print("NGC-Learn successfully installed!")
```

## <b>Citation:</b>

If you use ngc-learn in your research, please cite:
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

## <b>Contributing</b>

We welcome contributions! Please refer to the [contributing guidelines](CONTRIBUTING.md) for details.

## <b>License</b>
ngc-learn is distributed under the BSD 3-Clause License.
