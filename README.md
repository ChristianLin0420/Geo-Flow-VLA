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
│                           Geo-Flow VLA                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐                                  │
│  │   RGB Image  │────▶│  DINOv2-G/14 │────▶ Semantic Tokens (1536-dim)  │
│  │              │     │   (Frozen)   │                 │                │
│  └──────────────┘     └──────────────┘                 ▼                │
│         │                                    ┌──────────────────┐       │
│         │                                    │  Cross-Attention │       │
│         ▼                                    │     Fusion       │       │
│  ┌──────────────┐     ┌──────────────┐       └────────┬─────────┘       │
│  │   RGB Image  │────▶│    MoGe-2    │────▶ Geometric│Features          │
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
│  │                    DiT Policy (Flow Matching)                  │     │
│  │  ┌─────────────┐   ┌─────────────────────┐   ┌──────────────┐  │     │
│  │  │ Noisy Action│──▶│ DiT Blocks (x12)    │──▶│ Velocity v_t │  │     │
│  │  │     x_t     │   │ AdaLN Conditioning  │   │              │  │     │
│  │  └─────────────┘   └─────────────────────┘   └──────────────┘  │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Option 1: Docker (Recommended for SLURM/GPU Clusters)

```bash
# Pull the pre-built Docker image
docker pull christianlin0420/geo-flow-vla:latest

# Run interactive container with GPU support
docker run -it --gpus all \
    -v /path/to/data:/workspace/Geo-Flow-VLA/data \
    -v /path/to/checkpoints:/workspace/Geo-Flow-VLA/checkpoints \
    christianlin0420/geo-flow-vla:latest

# Run training directly
docker run --gpus all \
    -v /path/to/data:/workspace/Geo-Flow-VLA/data \
    -v /path/to/checkpoints:/workspace/Geo-Flow-VLA/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    christianlin0420/geo-flow-vla:latest \
    ./scripts/train.sh phase1 --gpus all --epochs 100
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/your-org/Geo-Flow-VLA.git
cd Geo-Flow-VLA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

# Install MoGe-2 (optional, for full 3D lifting)
pip install git+https://github.com/microsoft/MoGe.git

# Install LIBERO benchmark (optional, for training data)
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

### Build Docker Image Locally

```bash
# Build the image
docker build -t geo-flow-vla:latest .

# Or with BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t geo-flow-vla:latest .
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

### 2. Download CALVIN Dataset

```bash
# List available CALVIN datasets
python -m geo_flow_vla.data.download.download_calvin --list

# Download debug dataset for testing (~0.1 GB)
python -m geo_flow_vla.data.download.download_calvin --env debug --output ./data/calvin

# Download Task D environment (~35 GB, 369k samples)
python -m geo_flow_vla.data.download.download_calvin --env D --output ./data/calvin

# Download Tasks A, B, C environments (~105 GB, 1.14M samples)
python -m geo_flow_vla.data.download.download_calvin --env ABC --output ./data/calvin

# Download all environments ABCD (~140 GB, 1.42M samples)
python -m geo_flow_vla.data.download.download_calvin --env ABCD --output ./data/calvin

# Force re-download (resume interrupted downloads)
python -m geo_flow_vla.data.download.download_calvin --env D --output ./data/calvin --force

# Verify existing download
python -m geo_flow_vla.data.download.download_calvin --env D --output ./data/calvin --verify
```

Available environments:
- `debug` - Debug dataset for testing (~0.1 GB)
- `D` - Task D environment (369k samples, ~35 GB)
- `ABC` - Tasks A, B, C environments (1.14M samples, ~105 GB)
- `ABCD` - All environments (1.42M samples, ~140 GB)

**Note:** CALVIN uses LeRobot parquet format. To train with CALVIN, use the `calvin_config.yaml`:
```bash
./scripts/train.sh phase1 --config calvin_config --gpus all
```

### 3. Download RLBench Dataset

```bash
# List available RLBench tasks
python -m geo_flow_vla.data.download.download_rlbench --list

# Download all 18 tasks for all splits (~150 GB total)
python -m geo_flow_vla.data.download.download_rlbench --task all --split all --output ./data/rlbench

# Download only training split for all tasks
python -m geo_flow_vla.data.download.download_rlbench --task all --split train --output ./data/rlbench

# Download easy tasks only
python -m geo_flow_vla.data.download.download_rlbench --task easy --output ./data/rlbench

# Download medium tasks only
python -m geo_flow_vla.data.download.download_rlbench --task medium --output ./data/rlbench

# Download hard tasks only
python -m geo_flow_vla.data.download.download_rlbench --task hard --output ./data/rlbench

# Download specific tasks
python -m geo_flow_vla.data.download.download_rlbench --task stack_blocks push_buttons open_drawer --output ./data/rlbench

# Force re-download
python -m geo_flow_vla.data.download.download_rlbench --task all --force --output ./data/rlbench

# Download without extracting zip files
python -m geo_flow_vla.data.download.download_rlbench --task all --no-extract --output ./data/rlbench

# Verify existing download
python -m geo_flow_vla.data.download.download_rlbench --verify --output ./data/rlbench
```

Available task categories:
- `easy` - 6 tasks: close_jar, light_bulb_in, open_drawer, push_buttons, put_money_in_safe, turn_tap
- `medium` - 6 tasks: insert_onto_square_peg, meat_off_grill, place_cups, put_groceries_in_cupboard, stack_blocks, stack_cups
- `hard` - 6 tasks: place_shape_in_shape_sorter, place_wine_at_rack_location, put_item_in_drawer, reach_and_drag, slide_block_to_color_target, sweep_to_dustpan_of_size
- `all` - All 18 tasks

Available splits:
- `train` - 100 episodes per task
- `val` - 25 episodes per task
- `test` - 25 episodes per task

**Source:** [hqfang/RLBench-18-Tasks](https://huggingface.co/datasets/hqfang/RLBench-18-Tasks) on HuggingFace

**Note:** RLBench uses 8D actions (pos + quaternion + gripper). To train with RLBench, use the `rlbench_config.yaml`:
```bash
# Train on all tasks
./scripts/train.sh phase1 --config rlbench_config --gpus all

# Train on easy tasks only
./scripts/train.sh phase1 --config rlbench_config --gpus all data.rlbench_tasks=easy

# Train on specific tasks
./scripts/train.sh phase1 --config rlbench_config --gpus all "data.rlbench_tasks=[push_buttons,open_drawer]"
```

### 4. Phase 1: Train World Model

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

### 5. Phase 2: Train Policy

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

# Additional options
./scripts/train.sh phase1 --gpus 0,1,2,3 \
    --batch-size 64 \
    --epochs 100 \
    --workers 4 \
    --name "my_experiment" \
    --project "geo-flow-vla"
```

### 6. Evaluation

Evaluate trained models on all three benchmarks.

#### Checkpoint Loading Options

The evaluation scripts support two ways to specify checkpoints:

1. **Explicit paths (recommended)**: Directly specify world model and policy paths
2. **Legacy directory mode**: Provide a checkpoint directory with `phase1/` and `phase2/` subdirectories

#### Quick Start

```bash
# Explicit paths (recommended)
python -m geo_flow_vla.eval.eval_libero \
    --world_model_path ./checkpoints/phase1/world_model.pth \
    --policy_path ./checkpoints/phase2/best.pt \
    --suite libero_10

# Legacy mode (deprecated but supported)
python -m geo_flow_vla.eval.eval_libero \
    --checkpoint ./checkpoints/libero/full \
    --suite libero_10
```

#### LIBERO Evaluation

```bash
# Explicit paths (recommended)
python -m geo_flow_vla.eval.eval_libero \
    --world_model_path ./checkpoints/phase1/world_model.pth \
    --policy_path ./checkpoints/phase2/best.pt \
    --suite libero_10 \
    --n_rollouts 50

# Legacy checkpoint directory mode
python -m geo_flow_vla.eval.eval_libero \
    --checkpoint ./checkpoints/libero/full \
    --suite libero_10 \
    --n_rollouts 50
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--world_model_path` | None | Direct path to world model checkpoint (e.g., `phase1/world_model.pth`) |
| `--policy_path` | None | Direct path to policy checkpoint (e.g., `phase2/best.pt`) |
| `--checkpoint` | None | [DEPRECATED] Path to checkpoint directory with phase1/phase2 structure |
| `--config` | None | Path to config yaml (optional) |
| `--suite` | `libero_10` | Task suite: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, `libero_goal` |
| `--n_rollouts` | 50 | Number of rollouts per task |
| `--max_steps` | 400 | Maximum steps per episode |
| `--device` | `cuda` | Torch device |
| `--seed` | 42 | Random seed |
| `--no_wandb` | False | Disable W&B logging |
| `--save_videos` | False | Save rollout videos |
| `--output_dir` | `./eval_results/libero` | Output directory |

#### RLBench Evaluation

```bash
# Explicit paths (recommended)
python -m geo_flow_vla.eval.eval_rlbench \
    --world_model_path ./checkpoints/phase1/world_model.pth \
    --policy_path ./checkpoints/phase2/best.pt \
    --category all \
    --n_rollouts 25

# Legacy checkpoint directory mode
python -m geo_flow_vla.eval.eval_rlbench \
    --checkpoint ./checkpoints/rlbench/all \
    --category all \
    --n_rollouts 25
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--world_model_path` | None | Direct path to world model checkpoint |
| `--policy_path` | None | Direct path to policy checkpoint |
| `--checkpoint` | None | [DEPRECATED] Path to checkpoint directory |
| `--config` | None | Path to config yaml (optional) |
| `--tasks` | None | Specific tasks to evaluate (space-separated) |
| `--category` | None | Task category: `easy`, `medium`, `hard`, `all` |
| `--n_rollouts` | 25 | Number of rollouts per task |
| `--max_steps` | 200 | Maximum steps per episode |
| `--camera` | `front_rgb` | Camera view: `front_rgb`, `left_shoulder_rgb`, `right_shoulder_rgb`, `wrist_rgb` |
| `--device` | `cuda` | Torch device |
| `--seed` | 42 | Random seed |
| `--no_headless` | False | Show CoppeliaSim GUI |
| `--no_wandb` | False | Disable W&B logging |
| `--save_videos` | False | Save rollout videos |
| `--output_dir` | `./eval_results/rlbench` | Output directory |

**Task Categories:**
- `easy`: reach_target, pick_up_cup, push_buttons, pick_and_lift
- `medium`: take_lid_off_saucepan, put_item_in_drawer, stack_wine, put_knife_on_chopping_board, stack_blocks, close_jar, slide_block_to_color_target, meat_off_grill
- `hard`: place_wine_at_rack_location, place_cups, take_umbrella_out_of_umbrella_stand, sweep_to_dustpan_of_size, light_bulb_in, put_groceries_in_cupboard

#### CALVIN Evaluation

```bash
# Explicit paths (recommended)
python -m geo_flow_vla.eval.eval_calvin \
    --world_model_path ./checkpoints/phase1/world_model.pth \
    --policy_path ./checkpoints/phase2/best.pt \
    --calvin_root ./data/calvin \
    --split D \
    --n_chains 1000

# Legacy checkpoint directory mode
python -m geo_flow_vla.eval.eval_calvin \
    --checkpoint ./checkpoints/calvin/abc \
    --calvin_root ./data/calvin \
    --split D \
    --n_chains 1000
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--world_model_path` | None | Direct path to world model checkpoint |
| `--policy_path` | None | Direct path to policy checkpoint |
| `--checkpoint` | None | [DEPRECATED] Path to checkpoint directory |
| `--config` | None | Path to config yaml (optional) |
| `--calvin_root` | (required) | Path to CALVIN dataset/environment |
| `--split` | `D` | Evaluation split: `D`, `ABC`, `ABCD` |
| `--n_chains` | 1000 | Number of evaluation chains |
| `--tasks_per_chain` | 5 | Maximum tasks per chain |
| `--steps_per_task` | 360 | Maximum steps per subtask |
| `--device` | `cuda` | Torch device |
| `--seed` | 42 | Random seed |
| `--no_wandb` | False | Disable W&B logging |
| `--save_videos` | False | Save rollout videos |
| `--output_dir` | `./eval_results/calvin` | Output directory |

#### SLURM Batch Evaluation

```bash
# LIBERO (configure via environment variables)
SUITE=libero_10 CHECKPOINT=./checkpoints/libero/full sbatch sbatch_eval_libero.sh

# RLBench
CATEGORY=easy CHECKPOINT=./checkpoints/rlbench/all sbatch sbatch_eval_rlbench.sh

# CALVIN
SPLIT=D CHECKPOINT=./checkpoints/calvin/abc sbatch sbatch_eval_calvin.sh
```

#### Evaluation Metrics

| Benchmark | Primary Metric | Additional Metrics |
|-----------|---------------|-------------------|
| **LIBERO** | Success Rate (%) | Per-task success rate, avg steps to success |
| **RLBench** | Success Rate (%) | Per-task success rate, avg steps |
| **CALVIN** | Avg Completed Tasks (0-5) | % completing ≥1/2/3/4/5 tasks, full chain success rate |

Results are saved to JSON files in the output directory and optionally logged to W&B.

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
