ARG CUDA_VERSION="11.7.0"
ARG CUDNN_VERSION="8"
ARG OS_VERSION="22.04"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${OS_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------- packages -------------------------------- #

SHELL ["/bin/bash", "-c"]
ARG PYTHON_VERSION="3.10"
ARG PYTHON_MAJOR="3"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        vim \
        htop \
        iotop \
        git \
        git-lfs \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_MAJOR}-pip \
        python${PYTHON_MAJOR}-setuptools \
        python${PYTHON_MAJOR}-wheel \
        python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------- nvtop ---------------------------------- #

ARG OS_VERSION
RUN if [[ ${OS_VERSION} > "18.04" ]] ; then \
    apt-get update && \
    apt-get install -y --no-install-recommends nvtop ; else \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libncurses5-dev \
        libncursesw5-dev && \
    git clone https://github.com/Syllo/nvtop.git && \
    mkdir -p nvtop/build && \
    cd nvtop/build && \
    cmake .. && \
    make && \
    make install && \
    cd ../../ && \
    rm -rf nvtop ; fi

# ------------------------------ python checks ------------------------------ #

ENV PYTHONUNBUFFERED=1
RUN python3 --version
RUN pip3 --version

# ------------------------------- user & group ------------------------------ #

ARG USER_ID
ARG GROUP_ID
ARG NAME
RUN groupadd --gid ${GROUP_ID} ${NAME}
RUN useradd \
    --no-log-init \
    --create-home \
    --uid ${USER_ID} \
    --gid ${GROUP_ID} \
    -s /bin/sh ${NAME}

ARG WORKDIR_PATH
WORKDIR ${WORKDIR_PATH}

# ------------------------------- requirements ------------------------------ #

RUN mkdir /package
COPY . /package
# The --root-user-action option is available as of pip v22.1.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install \
        --no-cache-dir \
        --root-user-action ignore \
        -r /package/requirements.txt && \
    python3 -m pip install --no-cache-dir --root-user-action ignore /package
