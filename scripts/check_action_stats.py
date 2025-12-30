#!/usr/bin/env python3
"""
Check action statistics from training data.

This script helps debug action distribution mismatches between
training data and evaluation.

Usage:
    python scripts/check_action_stats.py
"""

import h5py
import numpy as np
from pathlib import Path
import glob


def check_libero_actions(data_root: str = "./data/libero"):
    """Check action statistics from LIBERO HDF5 files."""
    data_root = Path(data_root)
    
    all_actions = []
    
    # Find all demo files
    demo_files = list(data_root.glob("**/*demo*.hdf5"))
    
    if not demo_files:
        print(f"No demo files found in {data_root}")
        return
    
    print(f"Found {len(demo_files)} demo files")
    print("=" * 60)
    
    # Sample from first few files
    for demo_path in demo_files[:10]:
        try:
            with h5py.File(demo_path, 'r') as f:
                for demo_key in list(f['data'].keys())[:5]:  # First 5 demos per file
                    actions = f['data'][demo_key]['actions'][:]
                    all_actions.append(actions)
        except Exception as e:
            print(f"Error reading {demo_path}: {e}")
    
    if not all_actions:
        print("No actions loaded")
        return
    
    # Concatenate all actions
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"\n📊 LIBERO Action Statistics ({len(all_actions)} samples)")
    print("=" * 60)
    print(f"Overall: mean={all_actions.mean():.4f}, std={all_actions.std():.4f}")
    print(f"         min={all_actions.min():.4f}, max={all_actions.max():.4f}")
    
    print(f"\nPer-dimension statistics (action_dim={all_actions.shape[-1]}):")
    print("-" * 60)
    
    dim_names = [
        "x_delta", "y_delta", "z_delta",
        "rx_delta", "ry_delta", "rz_delta",
        "gripper"
    ]
    
    for i in range(all_actions.shape[-1]):
        dim_data = all_actions[:, i]
        name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        print(f"  {name:10s}: mean={dim_data.mean():8.4f}, std={dim_data.std():8.4f}, "
              f"range=[{dim_data.min():8.4f}, {dim_data.max():8.4f}]")
    
    print("\n" + "=" * 60)
    print("Compare these values with the [Action Debug] logs during evaluation!")
    print("If the ranges differ significantly, there may be a normalization mismatch.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/libero",
                        help="Path to LIBERO data directory")
    args = parser.parse_args()
    
    check_libero_actions(args.data_root)

