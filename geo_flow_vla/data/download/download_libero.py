"""
LIBERO Dataset Download Script.

Downloads the LIBERO benchmark data from HuggingFace.

Usage:
    python -m geo_flow_vla.data.download.download_libero --suite all --output ./data/libero

Reference:
    LIBERO GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO
    HuggingFace datasets: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
    HuggingFace fork: https://github.com/huggingface/lerobot-libero
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# HuggingFace dataset repository with original HDF5 format
# https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
LIBERO_HF_REPO = "yifengzhu-hf/LIBERO-datasets"

# Suite definitions
SUITE_DEFINITIONS = {
    "libero_spatial": {
        "prefix": "libero_spatial/",
        "num_tasks": 10,
        "description": "Tasks requiring spatial reasoning",
    },
    "libero_object": {
        "prefix": "libero_object/",
        "num_tasks": 10,
        "description": "Tasks requiring object knowledge transfer",
    },
    "libero_goal": {
        "prefix": "libero_goal/",
        "num_tasks": 10,
        "description": "Tasks requiring goal-oriented reasoning",
    },
    "libero_10": {
        "prefix": "libero_10/",
        "num_tasks": 10,
        "description": "10 test tasks for lifelong learning evaluation",
    },
    "libero_90": {
        "prefix": "libero_90/",
        "num_tasks": 90,
        "description": "90 pretraining tasks",
    },
}

SUITE_MAPPING = {
    "spatial": ["libero_spatial"],
    "object": ["libero_object"],
    "goal": ["libero_goal"],
    "10": ["libero_10"],
    "90": ["libero_90"],
    "100": ["libero_90", "libero_10"],
    "all": ["libero_spatial", "libero_object", "libero_goal"],
    "full": ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
}


def get_suite_files(suite_name: str) -> List[str]:
    """Get list of files for a specific suite from HuggingFace."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return []
    
    prefix = SUITE_DEFINITIONS.get(suite_name, {}).get("prefix", f"{suite_name}/")
    
    try:
        all_files = list_repo_files(LIBERO_HF_REPO, repo_type="dataset")
        suite_files = [f for f in all_files if f.startswith(prefix) and f.endswith('.hdf5')]
        return suite_files
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return []


def download_suite(
    suite_name: str,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """Download a single LIBERO suite from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    suite_output = output_dir / suite_name
    suite_output.mkdir(parents=True, exist_ok=True)
    
    files = get_suite_files(suite_name)
    if not files:
        logger.error(f"No files found for suite {suite_name}")
        return False
    
    logger.info(f"Downloading {len(files)} files for {suite_name}...")
    
    downloaded = 0
    for file_path in files:
        filename = Path(file_path).name
        local_path = suite_output / filename
        
        if local_path.exists() and not force:
            logger.debug(f"Skipping existing: {filename}")
            downloaded += 1
            continue
        
        try:
            hf_hub_download(
                repo_id=LIBERO_HF_REPO,
                filename=file_path,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            downloaded += 1
            logger.info(f"Downloaded: {filename} ({downloaded}/{len(files)})")
        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
    
    logger.info(f"Downloaded {downloaded}/{len(files)} files for {suite_name}")
    return downloaded == len(files)


def download_libero(
    output_dir: str = "./data/libero",
    suite: str = "all",
    force: bool = False,
) -> None:
    """
    Download LIBERO benchmark dataset from HuggingFace.
    
    Args:
        output_dir: Output directory for downloaded data
        suite: Which suite to download (spatial, object, goal, 10, 90, 100, all, full)
        force: Force re-download even if data exists
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if suite not in SUITE_MAPPING:
        raise ValueError(f"Unknown suite '{suite}'. Available: {list(SUITE_MAPPING.keys())}")
    
    suites_to_download = SUITE_MAPPING[suite]
    
    logger.info(f"Will download {len(suites_to_download)} suites: {suites_to_download}")
    logger.info(f"Output directory: {output_path}")
    
    results = {}
    for suite_name in suites_to_download:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {suite_name}...")
        logger.info(f"{'='*50}")
        
        success = download_suite(suite_name, output_path, force)
        results[suite_name] = success
    
    # Print summary
    print("\n" + "=" * 60)
    print("LIBERO Dataset Download Summary")
    print("=" * 60)
    
    for suite_name, success in results.items():
        suite_path = output_path / suite_name
        if suite_path.exists():
            num_files = len(list(suite_path.glob("*.hdf5")))
            status = f"✓ OK ({num_files} files)"
        else:
            status = "✗ FAILED"
        print(f"  {suite_name}: {status}")
    
    print("=" * 60)
    print(f"Data location: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download LIBERO dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suite options:
  spatial  - LIBERO-Spatial (10 tasks, spatial reasoning)
  object   - LIBERO-Object (10 tasks, object knowledge)
  goal     - LIBERO-Goal (10 tasks, goal-oriented)
  10       - LIBERO-10 (10 test tasks for lifelong learning)
  90       - LIBERO-90 (90 pretraining tasks)
  100      - LIBERO-100 (90 + 10 tasks)
  all      - All evaluation suites (spatial + object + goal)
  full     - Everything (spatial + object + goal + 10 + 90)

Examples:
  python -m geo_flow_vla.data.download.download_libero --suite spatial
  python -m geo_flow_vla.data.download.download_libero --suite all --output ./data/libero
        """
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=list(SUITE_MAPPING.keys()),
        help="Which suite to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/libero",
        help="Output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    
    args = parser.parse_args()
    
    download_libero(
        output_dir=args.output,
        suite=args.suite,
        force=args.force,
    )


if __name__ == "__main__":
    main()
