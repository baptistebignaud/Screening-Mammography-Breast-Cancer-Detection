ARG UBUNTU_VERSION=20.04
ARG CUDA=11.2
#FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3
FROM nvcr.io/nvidia/pytorch:22.12-py3
# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-devel-ubuntu${UBUNTU_VERSION} as base
# CMD nvidia-smi
ARG PYTHON_VERSION=3.8
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    python3.8-dev\
    unixodbc-dev\
    build-essential\
    graphviz\
    ffmpeg libsm6 libxext6\
    # Change default python
    && cd /usr/bin \
    && ln -sf python${PYTHON_VERSION}         python3 \
    && ln -sf python${PYTHON_VERSION}m        python3m \
    && ln -sf python${PYTHON_VERSION}-config  python3-config \
    && ln -sf python${PYTHON_VERSION}m-config python3m-config \
    && ln -sf python3                         /usr/bin/python \
    # Update pip and add common packages
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade \
    setuptools \
    wheel \
    six \
    # Cleanup
    && apt-get clean \
    && rm -rf $HOME/.cache/pip
# Download from the source
RUN curl -fsSL --silent https://deb.nodesource.com/setup_14.x |  bash 
RUN apt-get install nodejs -y
#RUN apt-get install npm -y
#RUN pip3 install --force-reinstall torch torchvision torchaudio

# next line cf. https://forums.developer.nvidia.com/t/what-is-cannot-allocate-memory-in-static-tls-block/169225
RUN export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-d22c30c5.so.1

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN jupyter labextension install jupyterlab-chart-editor
WORKDIR /appli

