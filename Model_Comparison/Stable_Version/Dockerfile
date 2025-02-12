# ------------------------------------------------------------------------------
# Example: CUDA 11.8 + cuDNN 8 + Ubuntu 20.04,
# with TensorFlow 2.10 (which supports CUDA 11.x).
# ------------------------------------------------------------------------------
    FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

    ENV DEBIAN_FRONTEND=noninteractive

    # Add NVIDIA repository and keys properly
    RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && rm -f /etc/apt/sources.list.d/nvidia*.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get update
    
    # Install TensorRT and other packages
    RUN apt-get install -y \
    tzdata software-properties-common git curl build-essential \
    python3.8 python3.8-dev python3-pip \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean
    
    # 2) Ensure "python3" points to Python 3.8
    RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
    
    # 3) Upgrade pip
    RUN python3 -m pip install --upgrade pip
    
    # 4) Install GPU-compatible TensorFlow + other Python packages
    #    TF 2.10 is known to support CUDA 11.x. Adjust versions as needed.
    RUN python3 -m pip install --no-cache-dir \
    tensorflow==2.10.* \
    numpy>=1.21 \
    scikit-learn==0.24.2 \
    matplotlib==3.4.3 \
    networkx==2.6.3 \
    holoviews \
    protobuf==3.19.6

    
    # 5) Set the working directory
    WORKDIR /app/ngc-learn
    
    # 6) Copy your local ngc-learn directory into the Docker image
    COPY . /app/ngc-learn
    
    # 7) Install ngc-learn (adjust your package’s install command as needed)
    RUN python3 setup.py install
    
    # Default command
    CMD ["/bin/bash"]

    