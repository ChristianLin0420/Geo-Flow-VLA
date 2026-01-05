"""
Abstract base class for benchmark evaluation.

Provides common functionality for all benchmark evaluators:
- Policy loading and inference
- Results tracking and logging
- W&B integration
- Video saving
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, logging will be disabled")


class BaseEvaluator(ABC):
    """
    Abstract base class for robotics benchmark evaluation.
    
    Subclasses implement benchmark-specific environment setup and rollout logic.
    """
    
    def __init__(
        self,
        world_model_path: Optional[str] = None,
        policy_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,  # Deprecated, for backward compat
        config_path: Optional[str] = None,
        device: str = "cuda",
        n_rollouts: int = 50,
        max_steps: int = 400,
        seed: int = 42,
        log_wandb: bool = True,
        save_videos: bool = False,
        output_dir: str = "./eval_results",
        run_name: Optional[str] = None,
    ):
        """
        Args:
            world_model_path: Direct path to world model checkpoint (e.g., phase1/world_model.pth)
            policy_path: Direct path to policy checkpoint (e.g., phase2/policy.pth or phase2/best.pt)
            checkpoint_dir: [DEPRECATED] Path to checkpoint directory with phase1/phase2 structure
            config_path: Path to config yaml
            device: Torch device
            n_rollouts: Number of evaluation rollouts per task
            max_steps: Maximum steps per episode
            seed: Random seed
            log_wandb: Log results to W&B
            save_videos: Save rollout videos
            output_dir: Directory for results
            run_name: Custom wandb run name (default: auto-generated)
        """
        self.world_model_path = world_model_path
        self.policy_path = policy_path
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.config_path = config_path
        self.device = device
        self.n_rollouts = n_rollouts
        self.max_steps = max_steps
        self.seed = seed
        self.log_wandb = log_wandb and WANDB_AVAILABLE
        self.save_videos = save_videos
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        
        # Set random seed
        np.random.seed(seed)
        
        # Load policy (lazy - done in setup)
        self.policy = None
        
        # Results storage
        self.results: Dict[str, List[Dict]] = {}
        
    def _init_policy(self) -> None:
        """Initialize policy wrapper."""
        from .policy_wrapper import GeoFlowVLAPolicy
        
        self.policy = GeoFlowVLAPolicy(
            world_model_path=self.world_model_path,
            policy_path=self.policy_path,
            checkpoint_dir=self.checkpoint_dir,
            config_path=self.config_path,
            device=self.device,
        )
        
        # Log checkpoint info
        if self.world_model_path and self.policy_path:
            logger.info(f"Policy loaded from explicit paths:")
            logger.info(f"  World model: {self.world_model_path}")
            logger.info(f"  Policy: {self.policy_path}")
        else:
            logger.info(f"Policy loaded from checkpoint_dir: {self.checkpoint_dir}")
    
    def _init_wandb(self) -> None:
        """Initialize wandb logging."""
        if self.log_wandb:
            # Build checkpoint info for logging
            if self.world_model_path and self.policy_path:
                checkpoint_info = f"{self.world_model_path}, {self.policy_path}"
            else:
                checkpoint_info = str(self.checkpoint_dir)
            
            wandb.init(
                project="geo-flow-vla-eval",
                name=self.run_name or f"{self.benchmark_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "world_model_path": str(self.world_model_path) if self.world_model_path else None,
                    "policy_path": str(self.policy_path) if self.policy_path else None,
                    "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
                    "n_rollouts": self.n_rollouts,
                    "max_steps": self.max_steps,
                    "benchmark": self.benchmark_name,
                    "seed": self.seed,
                }
            )
    
    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Return benchmark name (libero, rlbench, calvin)."""
        pass
    
    @abstractmethod
    def setup_environment(self, **kwargs) -> None:
        """Initialize benchmark environment."""
        pass
    
    @abstractmethod
    def get_tasks(self) -> List[str]:
        """Return list of task names to evaluate."""
        pass
    
    @abstractmethod
    def run_episode(
        self,
        task_name: str,
        episode_idx: int,
    ) -> Dict[str, Any]:
        """
        Run single evaluation episode.
        
        Returns:
            Dictionary with keys: success, steps, reward, video_frames (optional)
        """
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation across all tasks.
        
        Returns:
            Dictionary of metrics: {task_name: success_rate, ..., mean_success: float}
        """
        # Initialize policy if not done
        if self.policy is None:
            self._init_policy()
        
        # Initialize wandb
        self._init_wandb()
        
        tasks = self.get_tasks()
        print(f"\n{'='*60}")
        print(f"Evaluating {self.benchmark_name.upper()}: {len(tasks)} tasks × {self.n_rollouts} rollouts")
        print(f"{'='*60}\n")
        
        for task_idx, task_name in enumerate(tasks):
            print(f"\n▶ Task {task_idx+1}/{len(tasks)}: {task_name}")
            self.results[task_name] = []
            
            for ep_idx in range(self.n_rollouts):
                self.policy.reset()
                
                try:
                    result = self.run_episode(task_name, ep_idx)
                except Exception as e:
                    logger.error(f"Episode failed: {e}")
                    result = {
                        "success": False,
                        "steps": 0,
                        "reward": 0,
                        "error": str(e),
                    }
                
                self.results[task_name].append(result)
                
                status = "✓" if result.get("success", False) else "✗"
                print(f"  Episode {ep_idx+1}/{self.n_rollouts}: {status} ({result.get('steps', 0)} steps)")
                
                # Log to wandb
                if self.log_wandb:
                    wandb.log({
                        f"{task_name}/episode_success": int(result.get("success", False)),
                        f"{task_name}/episode_steps": result.get("steps", 0),
                        f"{task_name}/episode_reward": result.get("reward", 0),
                    })
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Log and save
        self._log_results(metrics)
        self._save_results(metrics)
        
        # Finish wandb
        if self.log_wandb:
            wandb.summary.update(metrics)
            wandb.finish()
        
        return metrics
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute success rates per task and overall."""
        metrics = {}
        all_successes = []
        
        for task_name, episodes in self.results.items():
            successes = [ep.get("success", False) for ep in episodes]
            success_rate = np.mean(successes) if successes else 0.0
            metrics[f"{task_name}/success_rate"] = success_rate
            
            # Average steps for successful episodes
            successful_steps = [ep["steps"] for ep in episodes if ep.get("success", False)]
            if successful_steps:
                metrics[f"{task_name}/avg_steps_to_success"] = np.mean(successful_steps)
            
            all_successes.extend(successes)
        
        metrics["mean_success_rate"] = np.mean(all_successes) if all_successes else 0.0
        metrics["total_episodes"] = len(all_successes)
        metrics["total_tasks"] = len(self.results)
        
        return metrics
    
    def _log_results(self, metrics: Dict[str, float]) -> None:
        """Log results to console and wandb."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.benchmark_name.upper()}")
        print(f"{'='*60}")
        
        # Per-task results
        for key, value in sorted(metrics.items()):
            if key.endswith("/success_rate"):
                task_name = key.replace("/success_rate", "")
                print(f"  {task_name}: {value*100:.1f}%")
        
        print(f"\n  📊 Mean Success Rate: {metrics['mean_success_rate']*100:.1f}%")
        print(f"  📊 Total Episodes: {metrics['total_episodes']}")
        print(f"{'='*60}\n")
        
        if self.log_wandb:
            wandb.log({"final_metrics": metrics})
    
    def _save_results(self, metrics: Dict[str, float]) -> None:
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"{self.benchmark_name}_{timestamp}.json"
        
        # Build checkpoint info
        if self.world_model_path and self.policy_path:
            checkpoint_info = {
                "world_model_path": str(self.world_model_path),
                "policy_path": str(self.policy_path),
            }
        else:
            checkpoint_info = {"checkpoint_dir": str(self.checkpoint_dir)}
        
        output = {
            "benchmark": self.benchmark_name,
            "checkpoint": checkpoint_info,
            "timestamp": timestamp,
            "seed": self.seed,
            "n_rollouts": self.n_rollouts,
            "max_steps": self.max_steps,
            "metrics": metrics,
            "episodes": {
                task: [
                    {k: v for k, v in ep.items() if k != "video_frames"}
                    for ep in episodes
                ]
                for task, episodes in self.results.items()
            },
        }
        
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {results_path}")
        
        if self.log_wandb:
            # Also upload to wandb
            wandb.save(str(results_path))
    
    def _save_video(self, task_name: str, episode_idx: int, frames: List[np.ndarray]) -> None:
        """Save episode video."""
        if not frames:
            return
        
        video_dir = self.output_dir / "videos" / task_name
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"episode_{episode_idx}.mp4"
        
        try:
            import imageio
            # Flip frames vertically for correct orientation in saved video
            flipped_frames = [np.flipud(frame) for frame in frames]
            imageio.mimsave(str(video_path), flipped_frames, fps=30)
            logger.info(f"Saved video: {video_path}")
            
            if self.log_wandb:
                # Use single key per task to avoid dashboard bloat (latest video overwrites)
                wandb.log({
                    f"{task_name}/video": wandb.Video(str(video_path), format="mp4")
                })
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")
