# Bipedal Leg Platform - Docker Container for H200 Training
# MJX (GPU-accelerated MuJoCo) + JAX PPO
#
# Build:  docker build -t bipedal-rl -f Dockerfile .
# Run:    docker run --gpus '"device=0"' -it --rm -v $(pwd):/workspace -w /workspace --shm-size=16g bipedal-rl bash

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure JAX sees the GPU
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget vim ffmpeg \
    libgl1-mesa-glx libglew-dev libosmesa6-dev \
    libglfw3-dev patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install JAX with CUDA first (order matters for compatibility)
RUN pip install --no-cache-dir "jax[cuda12]>=0.4.20"

# Install MuJoCo + MJX
RUN pip install --no-cache-dir "mujoco>=3.0.0" "mujoco-mjx>=3.0.0"

# Install Flax, Optax, Distrax for PPO
RUN pip install --no-cache-dir "flax>=0.8.0" "optax>=0.1.7" "distrax>=0.1.5"

# Install remaining dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.24.0" "tensorboard>=2.14.0" "matplotlib>=3.7.0" \
    "pyyaml>=6.0" "tqdm>=4.65.0" "imageio>=2.31.0" "imageio-ffmpeg>=0.4.8"

# SB3 fallback (CPU-based, for comparison/validation)
RUN pip install --no-cache-dir "gymnasium>=1.0.0" "stable-baselines3>=2.3.0" "torch>=2.1.0"

WORKDIR /workspace

# Verify installation
RUN python -c "\
import jax; \
import mujoco; \
from mujoco import mjx; \
print(f'JAX {jax.__version__}, devices: {jax.devices()}'); \
print(f'MuJoCo {mujoco.__version__}'); \
print('MJX available: OK')"

CMD ["bash"]
