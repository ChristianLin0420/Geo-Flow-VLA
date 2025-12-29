"""
RLBench Dataset Download Script.

Downloads the RLBench-18-Tasks benchmark dataset from HuggingFace Hub.

Source: https://huggingface.co/datasets/hqfang/RLBench-18-Tasks

Usage:
    python -m geo_flow_vla.data.download.download_rlbench --task all --output ./data/rlbench
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# RLBench-18-Tasks dataset on HuggingFace Hub
# Note: The repo name uses capital letters
RLBENCH_HF_REPO = "hqfang/RLBench-18-Tasks"

# 18 tasks available in the dataset (verified from HuggingFace repo)
RLBENCH_18_TASKS = [
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

# Task categories for convenience
RLBENCH_TASK_CATEGORIES = {
    "easy": [
        "close_jar",
        "light_bulb_in",
        "open_drawer",
        "push_buttons",
        "put_money_in_safe",
        "turn_tap",
    ],
    "medium": [
        "insert_onto_square_peg",
        "meat_off_grill",
        "place_cups",
        "put_groceries_in_cupboard",
        "stack_blocks",
        "stack_cups",
    ],
    "hard": [
        "place_shape_in_shape_sorter",
        "place_wine_at_rack_location",
        "put_item_in_drawer",
        "reach_and_drag",
        "slide_block_to_color_target",
        "sweep_to_dustpan_of_size",
    ],
    "all": RLBENCH_18_TASKS,
}

# Dataset splits available
RLBENCH_SPLITS = ["train", "val", "test"]


def download_rlbench_hf(
    output_dir: str = "./data/rlbench",
    tasks: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    force: bool = False,
    extract: bool = True,
) -> None:
    """
    Download RLBench-18-Tasks dataset from HuggingFace Hub.
    
    Args:
        output_dir: Output directory
        tasks: List of tasks to download (None = all 18 tasks)
        splits: List of splits to download (None = all splits)
        force: Force re-download
        extract: Extract zip files after download
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        logger.info("Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download, list_repo_files
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default to all tasks and splits
    if tasks is None:
        tasks = RLBENCH_18_TASKS
    if splits is None:
        splits = RLBENCH_SPLITS
    
    # Use local variable for repo name (may change if fallback is needed)
    repo_name = RLBENCH_HF_REPO
    
    logger.info(f"Downloading RLBench-18-Tasks from HuggingFace Hub")
    logger.info(f"Repository: {repo_name}")
    logger.info(f"Tasks: {len(tasks)} tasks")
    logger.info(f"Splits: {splits}")
    logger.info(f"Output: {output_path}")
    
    # Get list of files in the repository
    try:
        repo_files = list_repo_files(repo_name, repo_type="dataset")
        logger.info(f"Found {len(repo_files)} files in repository")
        
        # Debug: print first few files to understand structure
        zip_files = [f for f in repo_files if f.endswith('.zip')]
        logger.info(f"Found {len(zip_files)} zip files")
        if zip_files[:5]:
            logger.info(f"Sample zip files: {zip_files[:5]}")
            
    except Exception as e:
        logger.error(f"Failed to list repository files: {e}")
        logger.info("Trying alternative repository name...")
        # Try lowercase version
        try:
            repo_name = "hqfang/rlbench-18-tasks"
            repo_files = list_repo_files(repo_name, repo_type="dataset")
            logger.info(f"Using alternative repo: {repo_name}")
        except Exception as e2:
            logger.error(f"Failed with both repo names: {e2}")
            raise
    
    # Build a set of available files for quick lookup
    available_files = set(repo_files)
    
    # Download each task for each split
    downloaded_files = []
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for task in tasks:
            # The files are in format: data/split/task.zip
            zip_file_path = f"data/{split}/{task}.zip"
            
            if zip_file_path not in available_files:
                logger.warning(f"File not found: {zip_file_path}")
                continue
            
            local_zip_path = split_dir / f"{task}.zip"
            task_extract_path = split_dir / task
            
            # Check if already extracted
            if task_extract_path.exists() and not force:
                logger.info(f"✓ {split}/{task} already exists, skipping")
                continue
            
            # Download the zip file
            try:
                logger.info(f"Downloading {zip_file_path}...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_name,
                    filename=zip_file_path,
                    repo_type="dataset",
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                downloaded_files.append(downloaded_path)
                
                # Extract if requested
                if extract:
                    logger.info(f"Extracting to {split_dir}...")
                    with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                        zip_ref.extractall(split_dir)
                    
                logger.info(f"✓ {split}/{task} complete")
                
            except Exception as e:
                logger.error(f"Failed to download {split}/{task}: {e}")
                continue
    
    logger.info(f"RLBench download complete! Downloaded {len(downloaded_files)} files")
    if downloaded_files:
        verify_rlbench_hf(output_dir, tasks, splits)


def verify_rlbench_hf(
    data_dir: str,
    tasks: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> bool:
    """Verify RLBench HuggingFace dataset."""
    data_path = Path(data_dir)
    
    if tasks is None:
        tasks = RLBENCH_18_TASKS
    if splits is None:
        splits = RLBENCH_SPLITS
    
    all_valid = True
    
    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            logger.warning(f"✗ Missing split directory: {split}")
            all_valid = False
            continue
        
        for task in tasks:
            task_dir = split_dir / task
            
            if not task_dir.exists():
                # Check for zip file
                zip_path = split_dir / f"{task}.zip"
                if zip_path.exists():
                    logger.info(f"○ {split}/{task}: zip exists but not extracted")
                else:
                    logger.warning(f"✗ {split}/{task}: not found")
                    all_valid = False
                continue
            
            # Check for episode directories
            episode_dirs = list(task_dir.glob("episode*")) + list(task_dir.glob("variation*/episodes/episode*"))
            
            if episode_dirs:
                logger.info(f"✓ {split}/{task}: {len(episode_dirs)} episodes")
            else:
                # Check for all_variations directory structure
                all_var_dir = task_dir / "all_variations" / "episodes"
                if all_var_dir.exists():
                    episode_dirs = list(all_var_dir.glob("episode*"))
                    logger.info(f"✓ {split}/{task}: {len(episode_dirs)} episodes (all_variations)")
                else:
                    logger.warning(f"✗ {split}/{task}: no episodes found")
                    all_valid = False
    
    return all_valid


def list_available_tasks() -> None:
    """List all available RLBench tasks."""
    print("\nAvailable RLBench-18-Tasks:")
    print("=" * 60)
    
    for category, tasks in RLBENCH_TASK_CATEGORIES.items():
        if category == "all":
            continue
        print(f"\n{category.upper()} ({len(tasks)} tasks):")
        for task in tasks:
            print(f"  - {task}")
    
    print("\n" + "=" * 60)
    print(f"Total: {len(RLBENCH_18_TASKS)} tasks")
    print("\nSplits available:")
    print("  - train: 100 episodes per task")
    print("  - val: 25 episodes per task")
    print("  - test: 25 episodes per task")


def list_repo_contents() -> None:
    """List actual files in the HuggingFace repository."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import list_repo_files
    
    print(f"\nFiles in {RLBENCH_HF_REPO}:")
    print("=" * 60)
    
    try:
        files = list_repo_files(RLBENCH_HF_REPO, repo_type="dataset")
        zip_files = [f for f in files if f.endswith('.zip')]
        
        print(f"\nTotal files: {len(files)}")
        print(f"Zip files: {len(zip_files)}")
        
        print("\nZip files:")
        for f in sorted(zip_files):
            print(f"  {f}")
            
    except Exception as e:
        print(f"Error listing files: {e}")
        # Try alternative name
        try:
            alt_repo = "hqfang/rlbench-18-tasks"
            files = list_repo_files(alt_repo, repo_type="dataset")
            print(f"\nUsing alternative: {alt_repo}")
            zip_files = [f for f in files if f.endswith('.zip')]
            print(f"Zip files: {len(zip_files)}")
            for f in sorted(zip_files):
                print(f"  {f}")
        except Exception as e2:
            print(f"Error with alternative: {e2}")


def main():
    parser = argparse.ArgumentParser(description="Download RLBench-18-Tasks dataset from HuggingFace Hub")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to download (task names, 'easy', 'medium', 'hard', or 'all')",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=None,
        choices=RLBENCH_SPLITS + ["all"],
        help="Splits to download (train, val, test, or all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/rlbench",
        help="Output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract zip files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_tasks",
        help="List available tasks",
    )
    parser.add_argument(
        "--list-repo",
        action="store_true",
        dest="list_repo",
        help="List actual files in the HuggingFace repository",
    )
    
    args = parser.parse_args()
    
    if args.list_tasks:
        list_available_tasks()
        return
    
    if args.list_repo:
        list_repo_contents()
        return
    
    # Resolve task groups
    tasks = None
    if args.task:
        tasks = []
        for t in args.task:
            if t in RLBENCH_TASK_CATEGORIES:
                tasks.extend(RLBENCH_TASK_CATEGORIES[t])
            else:
                tasks.append(t)
        tasks = list(set(tasks))  # Remove duplicates
    
    # Resolve splits
    splits = args.split
    if splits and "all" in splits:
        splits = RLBENCH_SPLITS
    
    if args.verify:
        success = verify_rlbench_hf(args.output, tasks, splits)
        sys.exit(0 if success else 1)
    else:
        download_rlbench_hf(
            output_dir=args.output,
            tasks=tasks,
            splits=splits,
            force=args.force,
            extract=not args.no_extract,
        )


if __name__ == "__main__":
    main()
