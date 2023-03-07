# tags release notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
# tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?quick-deploy=false

ARG BASE_IMAGE=22.06-py3

FROM nvcr.io/nvidia/pytorch:${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    software-properties-common \
    cmake \
    g++-7 \
    git \
    gpg \
    curl \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    openssh-client \
    openssh-server \
    libsm6 \
    libxext6 \
    ffmpeg \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:${PATH}

# Create environment
ADD env_pytorch_nightly.yml /tmp/environment.yml
RUN /bin/bash -c "conda env create -f /tmp/environment.yml"
RUN /bin/bash -c "conda update -y conda"

ENV PATH /opt/conda/envs/pytorch-nightly/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch-nightly