# Geo-Flow VLA

**3D-Aware Robotic Foundation Policy with Conditional Flow Matching**

A research implementation targeting NeurIPS 2026, featuring a novel approach to robot policy learning that combines:

- **Dual Vision Encoders**: MoGe-2 (Geometric) + DINOv2-G (Semantic) for 3D-aware perception
- **DiT Policy Backbone**: Diffusion Transformer with Conditional Flow Matching
- **FB World Model**: Unsupervised Forward-Backward representation learning
- **CPR Regularization**: Conditional Policy Regularization for geometric consistency

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Geo-Flow VLA                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐                                  │
│  │   RGB Image  │────▶│  DINOv2-G/14 │────▶ Semantic Tokens (1536-dim)  │
│  │              │     │   (Frozen)   │                 │                │
│  └──────────────┘     └──────────────┘                 ▼                │
│         │                                    ┌──────────────────┐       │
│         │                                    │  Cross-Attention │       │
│         ▼                                    │     Fusion       │       │
│  ┌──────────────┐     ┌──────────────┐       └────────┬─────────┘       │
│  │   RGB Image  │────▶│    MoGe-2    │────▶ Geometric│Features         │
│  │              │     │   (Frozen)   │       (256-dim)│                 │
│  └──────────────┘     └──────────────┘                ▼                 │
│                                              ┌──────────────────┐       │
│                                              │  State Embedding │       │
│                                              │     (512-dim)    │       │
│                                              └────────┬─────────┘       │
│                                                       │                 │
│  ┌────────────────────────────────────────────────────┼─────────────┐   │
│  │                    FB World Model                  │             │   │
│  │  ┌───────────────┐                ┌───────────────┐│             │   │
│  │  │ Backward Net B │◀──── s_t ────▶│ Forward Net F ││             │   │
│  │  │  s → z ∈ S^d  │                │ (s,a,z)→s_t+k ││             │   │
│  │  └───────────────┘                └───────────────┘│             │   │
│  └────────────────────────────────────────────────────┼─────────────┘   │
│                                                       │                 │
│                                                       ▼                 │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    DiT Policy (Flow Matching)                   │     │
│  │  ┌─────────────┐   ┌─────────────────────┐   ┌──────────────┐  │     │
│  │  │ Noisy Action│──▶│ DiT Blocks (x12)   │──▶│ Velocity v_t │  │     │
│  │  │     x_t     │   │ AdaLN Conditioning  │   │              │  │     │
│  │  └─────────────┘   └─────────────────────┘   └──────────────┘  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/Geo-Flow-VLA.git
cd Geo-Flow-VLA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install MoGe-2 (optional, for full 3D lifting)
pip install git+https://github.com/microsoft/MoGe.git

# Install LIBERO benchmark (optional, for training data)
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

## Project Structure

```
geo_flow_vla/
├── configs/
│   └── config.yaml             # Hydra configuration
├── data/
│   ├── base_dataset.py         # Abstract base class for manipulation datasets
│   ├── libero_dataset.py       # LIBERO benchmark implementation
│   ├── rlbench_dataset.py      # RLBench placeholder
│   ├── calvin_dataset.py       # CALVIN placeholder
│   ├── moge_lifting.py         # MoGe-2 3D point map extraction
│   ├── normalizer.py           # State/action normalization
│   └── download/               # Dataset download scripts
│       ├── download_libero.py
│       ├── download_rlbench.py
│       └── download_calvin.py
├── models/
│   ├── dual_encoder.py         # DINOv2-G + MoGe-2 fusion
│   ├── world_model.py          # Forward-Backward networks
│   ├── diffusion_policy.py     # DiT with Flow Matching
│   └── discriminator.py        # CPR discriminator
├── losses/
│   ├── flow_matching_loss.py   # CFM objective
│   ├── fb_objective.py         # FB representation loss
│   └── cpr_regularizer.py      # Adversarial regularization
├── train/
│   ├── phase1_world_model.py   # Unsupervised FB pretraining
│   └── phase2_policy.py        # Policy training with CPR
├── utils/
│   ├── schedule.py             # Flow matching schedules
│   └── rotation_equivariance.py # SE(3) augmentations
└── __init__.py
```

## Quick Start

### 1. Download LIBERO Dataset

```bash
# Download all evaluation suites (~20GB for spatial, object, goal)
python -m geo_flow_vla.data.download.download_libero --suite all --output ./data/libero

# Or download everything including LIBERO-100 (~80GB)
python -m geo_flow_vla.data.download.download_libero --suite full --output ./data/libero
```

Available suites:
- `spatial` - LIBERO-Spatial (10 tasks)
- `object` - LIBERO-Object (10 tasks)
- `goal` - LIBERO-Goal (10 tasks)
- `10` - LIBERO-10 test tasks
- `90` - LIBERO-90 pretraining tasks
- `all` - spatial + object + goal (30 tasks)
- `full` - Everything (130 tasks)

### 2. Phase 1: Train World Model

```bash
# Single GPU training
python -m geo_flow_vla.train.phase1_world_model \
    training.phase1.epochs=100 \
    data.libero_suite=all

# Multi-GPU training (using train script)
./scripts/train.sh phase1 --gpus 0,1 --epochs 100

# Multi-GPU with torchrun directly
torchrun --nproc_per_node=2 -m geo_flow_vla.train.phase1_world_model \
    training.phase1.epochs=100 \
    data.libero_suite=all

# With WandB disabled for local testing
WANDB_MODE=disabled python -m geo_flow_vla.train.phase1_world_model \
    training.phase1.epochs=10 \
    data.libero_suite=spatial
```

### 3. Phase 2: Train Policy

```bash
# Single GPU
python -m geo_flow_vla.train.phase2_policy \
    training.phase2.epochs=200 \
    training.phase2.cpr_lambda_end=0.1

# Multi-GPU (4 GPUs on the PRO 6000 Blackwell cards)
./scripts/train.sh phase2 --gpus 2,3,4,5 --epochs 200
```

### Multi-GPU Training

The training scripts support distributed data parallel (DDP) training:

```bash
# Use the training launcher script
./scripts/train.sh phase1 --gpus 0,1        # Train on GPUs 0,1
./scripts/train.sh phase2 --gpus 2,3,4,5    # Train on GPUs 2-5
./scripts/train.sh phase1 --gpus all        # Use all available GPUs

# Available GPUs on this server:
#   0,1: NVIDIA RTX 6000 Ada (49GB each)
#   2,3,4,5: NVIDIA RTX PRO 6000 Blackwell (97GB each)
```

### 4. Evaluation

```bash
# Evaluate on LIBERO benchmark
python -m geo_flow_vla.eval.evaluate \
    --checkpoint ./checkpoints/phase2/policy.pth \
    --suite libero_spatial \
    --num_episodes 50
```

## Configuration

Key configuration options in `geo_flow_vla/configs/config.yaml`:

```yaml
model:
  state_dim: 512              # Fused state embedding
  action_dim: 7               # 7-DoF actions
  action_horizon: 16          # Action chunk size
  
  dit:
    hidden_dim: 768           # DiT hidden dimension
    num_layers: 12            # Transformer blocks
    num_heads: 12             # Attention heads
    
  fb:
    latent_dim: 256           # Goal embedding dimension
    ema_tau: 0.005            # EMA update rate

training:
  phase1:
    epochs: 100
    batch_size: 64
    lr: 1e-4
    
  phase2:
    epochs: 200
    batch_size: 32
    lr_policy: 3e-5
    cpr_lambda_end: 0.1
```

## WandB Logging

Training logs comprehensive metrics to Weights & Biases:

**Phase 1 (World Model):**
- FB losses (forward, backward, total)
- Embedding visualizations (t-SNE)
- Gradient norms

**Phase 2 (Policy):**
- Flow matching loss
- CPR discriminator accuracy
- Action MSE/MAE
- Generated trajectory visualizations

## Key Components

### Dual Encoder

Fuses semantic (DINOv2-G) and geometric (MoGe-2) features:

```python
from geo_flow_vla.models import DualEncoder

encoder = DualEncoder(
    semantic_dim=1536,    # DINOv2-G output
    geometric_dim=256,    # MoGe projection
    output_dim=512,       # Fused state
)

state = encoder(rgb_image)  # (B, 512)
```

### World Model

Forward-Backward representation learning:

```python
from geo_flow_vla.models import FBWorldModel

world_model = FBWorldModel(
    state_dim=512,
    action_dim=7,
    action_horizon=16,
    latent_dim=256,
)

# Get goal embedding
z = world_model.encode_goal(future_state)

# Predict future
pred_state = world_model.predict_future(state, actions, z)
```

### Diffusion Policy

DiT with Conditional Flow Matching:

```python
from geo_flow_vla.models import DiffusionPolicy

policy = DiffusionPolicy(
    action_dim=7,
    action_horizon=16,
    hidden_dim=768,
    num_layers=12,
)

# Training: compute CFM loss
loss = policy.compute_loss(gt_actions, state, goal)

# Inference: sample actions
actions = policy.sample(state, goal, num_steps=50)
```

## Citation

```bibtex
@inproceedings{geoflowvla2026,
  title={Geo-Flow VLA: 3D-Aware Robotic Foundation Policy with Conditional Flow Matching},
  author={Your Name},
  booktitle={NeurIPS},
  year={2026}
}
```

## References

- [MoGe](https://github.com/microsoft/MoGe) - Monocular Geometry Estimation (CVPR 2025)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised Vision Transformer
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - Lifelong Robot Learning Benchmark
- [LIBERO Datasets (HuggingFace)](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) - Dataset downloads
- [LeRobot-LIBERO](https://github.com/huggingface/lerobot-libero) - HuggingFace LIBERO fork
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - Visuomotor Policy Learning
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Generative Modeling with Optimal Transport

## License

MIT License - see [LICENSE](LICENSE) for details.
