"""
CALVIN Dataset Download Script.

Downloads the CALVIN benchmark dataset from official sources.

Usage:
    python -m geo_flow_vla.data.download.download_calvin --env D --output ./data/calvin

Reference:
    CALVIN Website: http://calvin.cs.uni-freiburg.de/
    GitHub: https://github.com/mees/calvin
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CALVIN dataset URLs
CALVIN_URLS = {
    "D": {
        "training": "http://calvin.cs.uni-freiburg.de/dataset/task_D_D/training.zip",
        "validation": "http://calvin.cs.uni-freiburg.de/dataset/task_D_D/validation.zip",
        "size_gb": 35.0,
    },
    "ABC": {
        "training": "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D/training.zip",
        "validation": "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D/validation.zip",
        "size_gb": 105.0,
    },
    "ABCD": {
        "training": "http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D/training.zip",
        "validation": "http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D/validation.zip",
        "size_gb": 140.0,
    },
    "debug": {
        "training": "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset/training.zip",
        "validation": "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset/validation.zip",
        "size_gb": 0.5,
    },
}


def download_with_wget(url: str, dest_path: Path) -> None:
    """Download file using wget."""
    cmd = ["wget", "-c", "-O", str(dest_path), url]
    subprocess.run(cmd, check=True)


def download_with_requests(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download file using requests with progress bar."""
    import requests
    from tqdm import tqdm
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    import zipfile
    
    logger.info(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def download_calvin(
    output_dir: str = "./data/calvin",
    env: str = "D",
    splits: Optional[List[str]] = None,
    force: bool = False,
    keep_zip: bool = False,
    use_wget: bool = True,
) -> None:
    """
    Download CALVIN dataset.
    
    Args:
        output_dir: Output directory
        env: Environment configuration ("D", "ABC", "ABCD", "debug")
        splits: Data splits to download (["training", "validation"] by default)
        force: Force re-download
        keep_zip: Keep zip files after extraction
        use_wget: Use wget for downloading (more reliable for large files)
    """
    if env not in CALVIN_URLS:
        raise ValueError(f"Unknown environment '{env}'. Available: {list(CALVIN_URLS.keys())}")
    
    if splits is None:
        splits = ["training", "validation"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    env_info = CALVIN_URLS[env]
    
    logger.info(f"Downloading CALVIN {env} (~{env_info['size_gb']} GB)")
    logger.info(f"Splits: {splits}")
    logger.info(f"Output: {output_path}")
    
    for split in splits:
        if split not in env_info:
            logger.warning(f"Split '{split}' not available for {env}")
            continue
        
        url = env_info[split]
        split_path = output_path / f"task_{env}_{env}" / split
        
        # Check if already exists
        if split_path.exists() and not force:
            logger.info(f"{split} already exists, skipping (use --force to re-download)")
            continue
        
        zip_path = output_path / f"{env}_{split}.zip"
        
        logger.info(f"Downloading {split}...")
        
        try:
            if use_wget:
                download_with_wget(url, zip_path)
            else:
                download_with_requests(url, zip_path)
            
            # Extract
            extract_zip(zip_path, output_path)
            
            if not keep_zip:
                zip_path.unlink()
                logger.info(f"Removed {zip_path}")
            
            logger.info(f"Successfully downloaded {split}")
            
        except Exception as e:
            logger.error(f"Failed to download {split}: {e}")
            logger.info(f"Please manually download from: {url}")
    
    logger.info("CALVIN download complete!")
    
    # Verify download
    verify_calvin(output_dir, env)


def verify_calvin(data_dir: str, env: str = "D") -> bool:
    """
    Verify CALVIN dataset.
    
    Args:
        data_dir: Path to CALVIN data
        env: Environment to verify
        
    Returns:
        True if verification passes
    """
    data_path = Path(data_dir)
    
    expected_dirs = {
        "D": ["task_D_D"],
        "ABC": ["task_ABC_D"],
        "ABCD": ["task_ABCD_D"],
        "debug": ["calvin_debug_dataset"],
    }
    
    all_valid = True
    
    for dir_name in expected_dirs.get(env, []):
        dir_path = data_path / dir_name
        
        if not dir_path.exists():
            logger.warning(f"Missing directory: {dir_path}")
            all_valid = False
            continue
        
        # Check for training and validation
        for split in ["training", "validation"]:
            split_path = dir_path / split
            
            if not split_path.exists():
                logger.warning(f"Missing split: {split_path}")
                all_valid = False
                continue
            
            # Check for key files
            npz_files = list(split_path.glob("*.npz"))
            
            if len(npz_files) > 0:
                logger.info(f"{split}: {len(npz_files)} episodes")
            else:
                logger.warning(f"{split}: No NPZ files found")
                all_valid = False
        
        # Check for language annotations
        lang_path = dir_path / "training" / "lang_annotations"
        if lang_path.exists():
            logger.info("Language annotations: OK")
        else:
            logger.warning("Language annotations not found")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Download CALVIN dataset")
    parser.add_argument(
        "--env",
        type=str,
        default="D",
        choices=list(CALVIN_URLS.keys()),
        help="Environment configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/calvin",
        help="Output directory",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["training", "validation"],
        help="Splits to download",
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
        "--keep-zip",
        action="store_true",
        help="Keep zip files",
    )
    parser.add_argument(
        "--no-wget",
        action="store_true",
        help="Use requests instead of wget",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_calvin(args.output, args.env)
        sys.exit(0 if success else 1)
    else:
        download_calvin(
            output_dir=args.output,
            env=args.env,
            splits=args.splits,
            force=args.force,
            keep_zip=args.keep_zip,
            use_wget=not args.no_wget,
        )


if __name__ == "__main__":
    main()

