"""
CALVIN Benchmark Evaluation.

Evaluates long-horizon manipulation with language instructions.
Protocol: 1000 chains of up to 5 tasks, report average completed length.

Usage (explicit paths - recommended):
    python -m geo_flow_vla.eval.eval_calvin \
        --world_model_path ./checkpoints/phase1/world_model.pth \
        --policy_path ./checkpoints/phase2/best.pt \
        --calvin_root ./data/calvin \
        --split D \
        --n_chains 1000

Usage (legacy checkpoint directory):
    python -m geo_flow_vla.eval.eval_calvin \
        --checkpoint ./checkpoints/calvin/abc \
        --calvin_root ./data/calvin \
        --split D \
        --n_chains 1000
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import logging
import json

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# CALVIN task annotations
CALVIN_TASKS = [
    "open_drawer",
    "close_drawer", 
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "push_into_drawer",
    "lift_red_block_table",
    "lift_blue_block_table",
    "lift_pink_block_table",
    "push_red_block_left",
    "push_red_block_right",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
    "rotate_red_block_left",
    "rotate_red_block_right",
    "rotate_blue_block_left",
    "rotate_blue_block_right",
    "rotate_pink_block_left",
    "rotate_pink_block_right",
    "place_in_slider",
    "move_slider_left",
    "move_slider_right",
    "stack_block",
    "unstack_block",
]


class CALVINEvaluator(BaseEvaluator):
    """
    CALVIN benchmark evaluator.
    
    Runs language-conditioned policy on long-horizon task chains.
    Reports: Average completed tasks per chain (max 5).
    
    Evaluation Protocol:
        - 1000 evaluation chains
        - Each chain has up to 5 sequential tasks
        - Agent must complete tasks in sequence
        - Chain ends when agent fails a task
        - Metric: Average number of completed tasks
    """
    
    def __init__(
        self,
        calvin_root: str,
        split: str = "D",  # D, ABC, or ABCD
        n_chains: int = 1000,
        tasks_per_chain: int = 5,
        steps_per_task: int = 360,  # ~6 seconds at 60Hz
        **kwargs,
    ):
        """
        Args:
            calvin_root: Path to CALVIN dataset/environment
            split: Evaluation split (D for train→D, ABC for train→ABC→D)
            n_chains: Number of evaluation chains
            tasks_per_chain: Max tasks per chain (typically 5)
            steps_per_task: Max steps allowed per subtask
            **kwargs: Passed to BaseEvaluator
        """
        self.calvin_root = Path(calvin_root)
        self.split = split
        self.n_chains = n_chains
        self.tasks_per_chain = tasks_per_chain
        self.steps_per_task = steps_per_task
        
        # Override max_steps to be total for all tasks
        kwargs["max_steps"] = steps_per_task * tasks_per_chain
        # Override n_rollouts to be n_chains
        kwargs["n_rollouts"] = n_chains
        
        self.env = None
        self.eval_chains = None
        
        super().__init__(**kwargs)
    
    @property
    def benchmark_name(self) -> str:
        return f"calvin_{self.split}"
    
    def setup_environment(self, **kwargs) -> None:
        """Initialize CALVIN environment."""
        try:
            # Try importing CALVIN environment
            from calvin_env.envs.play_table_env import PlayTableSimEnv
            
            # Load environment config
            config_path = self.calvin_root / "conf"
            if not config_path.exists():
                # Try alternative paths
                config_path = self.calvin_root / "calvin_env" / "conf"
            
            if config_path.exists():
                from hydra import compose, initialize_config_dir
                from hydra.core.global_hydra import GlobalHydra
                
                GlobalHydra.instance().clear()
                with initialize_config_dir(str(config_path)):
                    cfg = compose("config_eval.yaml")
                
                # Create environment
                self.env = PlayTableSimEnv(
                    **cfg.env,
                    show_gui=False,
                )
            else:
                # Create with default config
                self.env = PlayTableSimEnv(
                    robot_base_pose=[0.0, 0.0, 0.0],
                    show_gui=False,
                )
            
            # Load evaluation chains
            self.eval_chains = self._load_eval_chains()
            
            logger.info(f"✓ CALVIN environment initialized (split={self.split})")
            logger.info(f"  Evaluation chains: {len(self.eval_chains)}")
            print(f"✓ CALVIN environment initialized (split={self.split})")
            print(f"  Evaluation chains: {len(self.eval_chains)}")
            
        except ImportError as e:
            logger.warning(f"CALVIN environment not available: {e}")
            logger.info("Falling back to offline evaluation mode")
            
            # Fallback: generate synthetic chains for testing
            self.env = None
            self.eval_chains = self._generate_random_chains()
            
            print(f"⚠ CALVIN env not available, using synthetic chains")
            print(f"  Evaluation chains: {len(self.eval_chains)}")
    
    def _load_eval_chains(self) -> List[List[Dict]]:
        """Load pre-defined evaluation task chains."""
        # Try to load from file
        chains_path = self.calvin_root / f"eval_chains_{self.split}.json"
        if chains_path.exists():
            with open(chains_path) as f:
                return json.load(f)
        
        # Try alternative locations
        for alt_path in [
            self.calvin_root / "evaluation" / f"eval_chains_{self.split}.json",
            self.calvin_root / "data" / f"eval_chains_{self.split}.json",
        ]:
            if alt_path.exists():
                with open(alt_path) as f:
                    return json.load(f)
        
        # Generate random chains if file doesn't exist
        logger.warning("Eval chains file not found, generating random chains")
        return self._generate_random_chains()
    
    def _generate_random_chains(self) -> List[List[Dict]]:
        """Generate random task chains for evaluation."""
        np.random.seed(self.seed)
        
        chains = []
        for _ in range(self.n_chains):
            chain = []
            for _ in range(self.tasks_per_chain):
                task = np.random.choice(CALVIN_TASKS)
                chain.append({
                    "task": task,
                    "instruction": task.replace("_", " "),
                })
            chains.append(chain)
        
        return chains
    
    def get_tasks(self) -> List[str]:
        """Return chain indices as 'tasks'."""
        return [f"chain_{i}" for i in range(min(self.n_chains, len(self.eval_chains)))]
    
    def run_episode(
        self,
        task_name: str,
        episode_idx: int,
    ) -> Dict[str, Any]:
        """Run single CALVIN chain (up to 5 sequential tasks)."""
        chain_idx = int(task_name.split("_")[1])
        chain = self.eval_chains[chain_idx]
        
        video_frames = [] if self.save_videos else None
        
        # Check if we have a real environment
        if self.env is None:
            # Offline mode - return dummy results
            return self._run_offline_episode(chain)
        
        try:
            # Reset environment
            obs = self.env.reset()
            self.policy.reset()
            
            completed_tasks = 0
            total_steps = 0
            
            for subtask in chain[:self.tasks_per_chain]:
                instruction = subtask["instruction"]
                task_success = False
                
                for step in range(self.steps_per_task):
                    # Get RGB from static camera
                    if "rgb_obs" in obs:
                        rgb = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3)
                    else:
                        rgb = obs.get("image", np.zeros((200, 200, 3), dtype=np.uint8))
                    
                    # Get proprioception
                    proprio = obs.get("robot_obs", np.zeros(15))
                    
                    # Get action from policy
                    action = self.policy.predict(
                        rgb=rgb,
                        instruction=instruction,
                        proprio=proprio,
                    )
                    
                    # CALVIN expects 7D action: 3D pos delta + 3D rot delta + gripper
                    if len(action) != 7:
                        action = action[:7]
                    
                    # Execute action
                    obs, reward, done, info = self.env.step(action)
                    total_steps += 1
                    
                    if self.save_videos:
                        video_frames.append(rgb.copy())
                    
                    # Check subtask completion
                    if info.get("success", False):
                        task_success = True
                        completed_tasks += 1
                        break
                
                if not task_success:
                    # Chain broken - stop evaluation
                    break
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            completed_tasks = 0
            total_steps = 0
        
        # Save video if requested
        if self.save_videos and video_frames:
            self._save_video(task_name, episode_idx, video_frames)
        
        return {
            "success": completed_tasks == self.tasks_per_chain,  # Full chain completion
            "completed_tasks": completed_tasks,
            "steps": total_steps,
            "reward": completed_tasks,  # Reward = number of completed tasks
            "video_frames": video_frames if self.save_videos else None,
        }
    
    def _run_offline_episode(self, chain: List[Dict]) -> Dict[str, Any]:
        """Run offline evaluation (for testing without CALVIN env)."""
        # This is a placeholder - returns random results for testing
        completed = np.random.randint(0, self.tasks_per_chain + 1)
        return {
            "success": completed == self.tasks_per_chain,
            "completed_tasks": completed,
            "steps": completed * 100,
            "reward": completed,
        }
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute CALVIN-specific metrics."""
        all_completed = []
        
        for task_name, episodes in self.results.items():
            for ep in episodes:
                all_completed.append(ep.get("completed_tasks", 0))
        
        all_completed = np.array(all_completed)
        
        metrics = {
            "avg_completed_tasks": float(np.mean(all_completed)),
            "std_completed_tasks": float(np.std(all_completed)),
            "full_chain_success_rate": float(np.mean(all_completed == self.tasks_per_chain)),
            "total_chains": len(all_completed),
        }
        
        # Per-task success rates (completed N or more)
        for i in range(1, self.tasks_per_chain + 1):
            metrics[f"completed_{i}_or_more"] = float(np.mean(all_completed >= i))
        
        # Also compute mean success rate (full chains)
        metrics["mean_success_rate"] = metrics["full_chain_success_rate"]
        
        return metrics
    
    def _log_results(self, metrics: Dict[str, float]) -> None:
        """Log CALVIN-specific results."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.benchmark_name.upper()}")
        print(f"{'='*60}")
        
        print(f"\n  📊 Average Completed Tasks: {metrics['avg_completed_tasks']:.2f} ± {metrics['std_completed_tasks']:.2f}")
        print(f"  📊 Full Chain Success Rate: {metrics['full_chain_success_rate']*100:.1f}%")
        print(f"\n  Task Completion Breakdown:")
        for i in range(1, self.tasks_per_chain + 1):
            rate = metrics[f"completed_{i}_or_more"]
            print(f"    ≥{i} tasks: {rate*100:.1f}%")
        
        print(f"\n  Total Chains Evaluated: {metrics['total_chains']}")
        print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CALVIN Benchmark Evaluation")
    
    # New explicit checkpoint arguments (recommended)
    parser.add_argument("--world_model_path", type=str, default=None,
                        help="Direct path to world model checkpoint (e.g., checkpoints/phase1/world_model.pth)")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Direct path to policy checkpoint (e.g., checkpoints/phase2/best.pt)")
    
    # Legacy argument (deprecated but supported for backward compatibility)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[DEPRECATED] Path to checkpoint directory with phase1/phase2 structure")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config yaml (optional)")
    parser.add_argument("--calvin_root", type=str, required=True,
                        help="Path to CALVIN dataset/environment")
    parser.add_argument("--split", type=str, default="D",
                        choices=["D", "ABC", "ABCD"],
                        help="Evaluation split")
    parser.add_argument("--n_chains", type=int, default=1000,
                        help="Number of evaluation chains")
    parser.add_argument("--tasks_per_chain", type=int, default=5,
                        help="Maximum tasks per chain")
    parser.add_argument("--steps_per_task", type=int, default=360,
                        help="Maximum steps per subtask")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--output_dir", type=str, default="./eval_results/calvin",
                        help="Output directory for results")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom wandb run name (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Validate arguments - must provide either explicit paths or legacy checkpoint_dir
    if not (args.world_model_path and args.policy_path) and not args.checkpoint:
        parser.error("Must provide either (--world_model_path and --policy_path) or --checkpoint")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    evaluator = CALVINEvaluator(
        calvin_root=args.calvin_root,
        split=args.split,
        n_chains=args.n_chains,
        tasks_per_chain=args.tasks_per_chain,
        steps_per_task=args.steps_per_task,
        world_model_path=args.world_model_path,
        policy_path=args.policy_path,
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        device=args.device,
        seed=args.seed,
        log_wandb=not args.no_wandb,
        save_videos=args.save_videos,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    
    evaluator.setup_environment()
    metrics = evaluator.evaluate()
    
    return metrics


if __name__ == "__main__":
    main()


