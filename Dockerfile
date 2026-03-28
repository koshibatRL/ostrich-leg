# Bipedal Leg Platform - Docker Container for H200 Training
# Based on Isaac Lab's recommended Docker setup
#
# Build:  docker build -t bipedal-rl -f Dockerfile .
# Run:    docker run --gpus '"device=0"' -it --rm -v $(pwd):/workspace -w /workspace bipedal-rl bash

FROM nvcr.io/nvidia/isaac-sim:4.5.0 AS base

# Alternatively, if Isaac Lab pip install is preferred:
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Clone and install legged_gym + rsl_rl (core training framework)
WORKDIR /opt
RUN git clone https://github.com/leggedrobotics/legged_gym.git && \
    cd legged_gym && \
    pip install -e .

RUN git clone https://github.com/leggedrobotics/rsl_rl.git && \
    cd rsl_rl && \
    pip install -e .

# Clone unitree_rl_gym for reference configs
RUN git clone https://github.com/unitreerobotics/unitree_rl_gym.git

# MuJoCo menagerie for reference robot models
RUN git clone https://github.com/google-deepmind/mujoco_menagerie.git

WORKDIR /workspace

# Verify installation
RUN python -c "import mujoco; import torch; print(f'MuJoCo {mujoco.__version__}, PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

CMD ["bash"]
