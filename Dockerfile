# =============================================================================
# Geo-Flow VLA Docker Image
# 3D-Aware Robotic Foundation Policy with Conditional Flow Matching
# =============================================================================

# Base: PyTorch with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL maintainer="Geo-Flow VLA Team"
LABEL description="Geo-Flow VLA: 3D-aware robotic foundation policy with CFM backbone"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /workspace

# =============================================================================
# Install System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    # For headless rendering (MoGe/visualization)
    libegl1-mesa \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libglvnd-dev \
    # HDF5 for LIBERO dataset
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment for headless rendering
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# =============================================================================
# Upgrade pip and install build tools
# =============================================================================
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# Install Core Python Dependencies
# =============================================================================
# Copy requirements first for caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Install Vision Models
# =============================================================================

# Install transformers (required by MoGe-2)
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0

# Install MoGe-2 from source
RUN pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git

# Install HuggingFace Hub for model downloads
RUN pip install --no-cache-dir huggingface_hub>=0.23.0

# =============================================================================
# Install LIBERO Benchmark (for data loading)
# =============================================================================
RUN pip install --no-cache-dir git+https://github.com/Lifelong-Robot-Learning/LIBERO.git || \
    echo "Warning: LIBERO installation failed, continuing without it"

# =============================================================================
# Install Flash Attention (optional, for faster training)
# =============================================================================
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Warning: Flash Attention installation failed, continuing without it"

# =============================================================================
# Install xFormers (optional, for DINOv2 efficiency)
# =============================================================================
RUN pip install --no-cache-dir xformers || \
    echo "Warning: xFormers installation failed, continuing without it"

# =============================================================================
# Copy Project and Install
# =============================================================================
COPY . /workspace/Geo-Flow-VLA/

# Install the package in editable mode
WORKDIR /workspace/Geo-Flow-VLA
RUN pip install -e .

# =============================================================================
# Install Additional Tools
# =============================================================================
# WandB with media support for 3D point clouds
RUN pip install --no-cache-dir 'wandb[media]'

# =============================================================================
# Create Necessary Directories
# =============================================================================
RUN mkdir -p \
    /workspace/Geo-Flow-VLA/data/libero \
    /workspace/Geo-Flow-VLA/data/rlbench \
    /workspace/Geo-Flow-VLA/data/calvin \
    /workspace/Geo-Flow-VLA/checkpoints/phase1 \
    /workspace/Geo-Flow-VLA/checkpoints/phase2 \
    /workspace/Geo-Flow-VLA/logs \
    /workspace/Geo-Flow-VLA/outputs

# =============================================================================
# Set PYTHONPATH
# =============================================================================
ENV PYTHONPATH=/workspace/Geo-Flow-VLA

# =============================================================================
# Fix torch/torchvision versions (xformers may have upgraded them)
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Uninstall incompatible xformers (was built for wrong torch version)
# and reinstall compatible version for PyTorch 2.4.0
RUN pip uninstall -y xformers && \
    pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu124 || \
    echo "Warning: xFormers installation failed, DINOv2 will work without it"

# =============================================================================
# Pre-download Models (optional, comment out to reduce image size)
# =============================================================================
# Pre-download DINOv2-G/14 weights
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')" || \
    echo "Warning: DINOv2 pre-download failed"

# Pre-download MoGe-2 weights
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Ruicheng/moge-2-vitl-normal', 'model.pt')" || \
    echo "Warning: MoGe-2 pre-download failed"

# =============================================================================
# Verify Installation
# =============================================================================
RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" && \
    python -c "import einops; print('✓ Einops OK')" && \
    python -c "import hydra; print('✓ Hydra OK')" && \
    python -c "import wandb; print('✓ WandB OK')" && \
    python -c "import h5py; print('✓ H5py OK')" && \
    python -c "import cv2; print('✓ OpenCV OK')" && \
    python -c "from geo_flow_vla.models.dual_encoder import DualEncoder; print('✓ DualEncoder OK')" && \
    python -c "from geo_flow_vla.models.world_model import FBWorldModel; print('✓ WorldModel OK')" && \
    python -c "from geo_flow_vla.models.diffusion_policy import DiffusionPolicy; print('✓ DiffusionPolicy OK')" && \
    python -c "from geo_flow_vla.train.phase1_world_model import WorldModelTrainer; print('✓ Phase1 Trainer OK')" && \
    python -c "from geo_flow_vla.train.phase2_policy import PolicyTrainer; print('✓ Phase2 Trainer OK')"

# =============================================================================
# Expose Ports
# =============================================================================
# WandB UI (optional)
EXPOSE 8080

# =============================================================================
# Default Command
# =============================================================================
CMD ["/bin/bash"]

