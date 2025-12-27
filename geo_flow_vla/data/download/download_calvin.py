"""
CALVIN Dataset Download Script.

Downloads the CALVIN benchmark dataset from HuggingFace Hub (LeRobot format).

Source: https://huggingface.co/fywang

Usage:
    python -m geo_flow_vla.data.download.download_calvin --env D --output ./data/calvin
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CALVIN datasets on HuggingFace Hub (LeRobot format)
# Source: https://huggingface.co/fywang
CALVIN_HF_DATASETS = {
    "debug": {
        "repo_id": "fywang/calvin-debug-lerobot",
        "size_gb": 0.1,
        "description": "Debug dataset for testing",
    },
    "D": {
        "repo_id": "fywang/calvin-task-D-D-lerobot",
        "size_gb": 35.0,
        "description": "Task D environment (369k samples)",
    },
    "ABC": {
        "repo_id": "fywang/calvin-task-ABC-D-lerobot",
        "size_gb": 105.0,
        "description": "Tasks A, B, C environments (1.14M samples)",
    },
    "ABCD": {
        "repo_id": "fywang/calvin-task-ABCD-D-lerobot",
        "size_gb": 140.0,
        "description": "All environments (1.42M samples)",
    },
}


def download_calvin_hf(
    output_dir: str = "./data/calvin",
    env: str = "D",
    force: bool = False,
) -> None:
    """
    Download CALVIN dataset from HuggingFace Hub.
    
    Args:
        output_dir: Output directory
        env: Environment configuration ("D", "ABC", "ABCD", "debug")
        force: Force re-download
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.info("Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download
    
    if env not in CALVIN_HF_DATASETS:
        raise ValueError(f"Unknown environment '{env}'. Available: {list(CALVIN_HF_DATASETS.keys())}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    env_info = CALVIN_HF_DATASETS[env]
    repo_id = env_info["repo_id"]
    
    logger.info(f"Downloading CALVIN {env} from HuggingFace Hub")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Size: ~{env_info['size_gb']} GB")
    logger.info(f"Description: {env_info['description']}")
    logger.info(f"Output: {output_path}")
    
    # Target directory for this environment
    local_dir = output_path / f"calvin_{env.lower()}_lerobot"
    
    if local_dir.exists() and not force:
        logger.info(f"Data already exists at {local_dir}")
        logger.info("Use --force to re-download")
        verify_calvin_hf(output_dir, env)
        return
    
    try:
        # Download from HuggingFace Hub
        logger.info("Starting download...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        logger.info("CALVIN download complete!")
        verify_calvin_hf(output_dir, env)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(f"You can manually download from: https://huggingface.co/datasets/{repo_id}")
        raise


def verify_calvin_hf(data_dir: str, env: str = "D") -> bool:
    """Verify CALVIN HuggingFace dataset."""
    data_path = Path(data_dir)
    
    local_dir = data_path / f"calvin_{env.lower()}_lerobot"
    
    if not local_dir.exists():
        logger.warning(f"Missing directory: {local_dir}")
        return False
    
    # Check for LeRobot dataset files
    parquet_files = list(local_dir.glob("**/*.parquet"))
    arrow_files = list(local_dir.glob("**/*.arrow"))
    
    if parquet_files:
        logger.info(f"✓ Found {len(parquet_files)} parquet files")
    elif arrow_files:
        logger.info(f"✓ Found {len(arrow_files)} arrow files")
    else:
        logger.warning("✗ No data files found")
        return False
    
    # Check for metadata
    meta_files = list(local_dir.glob("**/meta*.json")) + list(local_dir.glob("**/*config*.json"))
    if meta_files:
        logger.info(f"✓ Found metadata files: {[f.name for f in meta_files[:3]]}")
    
    return True


def load_calvin_lerobot(data_dir: str, env: str = "D"):
    """
    Load CALVIN dataset using LeRobot format.
    
    Example usage:
        dataset = load_calvin_lerobot("./data/calvin", "D")
        for sample in dataset:
            image = sample["observation.image"]
            action = sample["action"]
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        logger.error("LeRobot not installed. Install with: pip install lerobot")
        return None
    
    repo_id = CALVIN_HF_DATASETS[env]["repo_id"]
    
    # Load directly from HuggingFace Hub
    dataset = LeRobotDataset(repo_id)
    
    logger.info(f"Loaded CALVIN {env} dataset with {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Download CALVIN dataset from HuggingFace Hub")
    parser.add_argument(
        "--env",
        type=str,
        default="D",
        choices=list(CALVIN_HF_DATASETS.keys()),
        help="Environment configuration (debug, D, ABC, ABCD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/calvin",
        help="Output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List available datasets",
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("\nAvailable CALVIN datasets on HuggingFace Hub:")
        print("=" * 60)
        for env, info in CALVIN_HF_DATASETS.items():
            print(f"\n{env}:")
            print(f"  Repository: {info['repo_id']}")
            print(f"  Size: ~{info['size_gb']} GB")
            print(f"  Description: {info['description']}")
        print("\n" + "=" * 60)
        return
    
    if args.verify:
        success = verify_calvin_hf(args.output, args.env)
        sys.exit(0 if success else 1)
    else:
        download_calvin_hf(
            output_dir=args.output,
            env=args.env,
            force=args.force,
        )


if __name__ == "__main__":
    main()