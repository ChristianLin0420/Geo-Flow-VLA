"""
LIBERO Benchmark Evaluation.

Evaluates on LIBERO task suites using MuJoCo simulation.
Suites: libero_10, libero_90, libero_spatial, libero_object, libero_goal

Usage:
    python -m geo_flow_vla.eval.eval_libero \
        --checkpoint ./checkpoints/libero/full \
        --suite libero_10 \
        --n_rollouts 50
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Available LIBERO task suites
LIBERO_SUITES = [
    "libero_10",
    "libero_90", 
    "libero_spatial",
    "libero_object",
    "libero_goal",
]


class LIBEROEvaluator(BaseEvaluator):
    """
    LIBERO benchmark evaluator.
    
    Runs policy rollouts in LIBERO MuJoCo environments
    and reports per-task success rates.
    """
    
    def __init__(
        self,
        task_suite: str = "libero_10",
        **kwargs,
    ):
        """
        Args:
            task_suite: One of libero_10, libero_90, libero_spatial, 
                        libero_object, libero_goal
            **kwargs: Passed to BaseEvaluator
        """
        if task_suite not in LIBERO_SUITES:
            raise ValueError(f"Unknown suite: {task_suite}. Choose from {LIBERO_SUITES}")
        
        self.task_suite = task_suite
        self.task_configs = None
        self._env_class = None
        
        super().__init__(**kwargs)
        
    @property
    def benchmark_name(self) -> str:
        return f"libero_{self.task_suite}"
    
    def setup_environment(self, **kwargs) -> None:
        """Initialize LIBERO environments."""
        try:
            # LIBERO has nested package structure: libero/libero/envs
            from libero.libero.envs import TASK_MAPPING
            from libero.libero import get_libero_path
            
            # Get benchmark directory
            import os
            import json
            
            benchmark_root = get_libero_path("benchmark_root")
            task_suite_name = self.task_suite if self.task_suite.startswith("libero_") else f"libero_{self.task_suite}"
            
            # Load task configs from benchmark JSON
            task_suite_file = os.path.join(benchmark_root, f"{task_suite_name}.json")
            
            if os.path.exists(task_suite_file):
                with open(task_suite_file, 'r') as f:
                    self.task_configs = json.load(f)
                self._tasks_from_json = True
            else:
                # Fallback: use TASK_MAPPING keys
                logger.warning(f"Task suite file not found: {task_suite_file}")
                self.task_configs = [{"name": k} for k in TASK_MAPPING.keys()]
                self._tasks_from_json = False
            
            logger.info(f"✓ LIBERO {self.task_suite}: {len(self.task_configs)} tasks")
            print(f"✓ LIBERO {self.task_suite}: {len(self.task_configs)} tasks")
            
        except ImportError as e:
            raise ImportError(
                f"LIBERO not installed: {e}\n"
                "Install with:\n"
                "  cd /tmp && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
                "  touch /tmp/LIBERO/libero/__init__.py\n"
                "  export PYTHONPATH=/tmp/LIBERO:$PYTHONPATH\n"
                "  pip install robosuite==1.4.1"
            )
    
    def get_tasks(self) -> List[str]:
        """Return task names in suite."""
        if self.task_configs is None:
            self.setup_environment()
        # Handle both dict format and object format
        tasks = []
        for task in self.task_configs:
            if isinstance(task, dict):
                tasks.append(task.get("name", task.get("task_name", str(task))))
            else:
                tasks.append(task.name if hasattr(task, 'name') else str(task))
        return tasks
    
    def _create_env(self, task_idx: int):
        """Create environment for a specific task."""
        from libero.libero.envs import OffScreenRenderEnv
        
        task = self.task_configs[task_idx]
        
        # Get bddl file path
        if isinstance(task, dict):
            bddl_file = task.get("bddl_file", task.get("bddl_file_name", ""))
        else:
            bddl_file = task.bddl_file if hasattr(task, 'bddl_file') else ""
        
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_names=["agentview"],
            camera_heights=[128],
            camera_widths=[128],
            reward_shaping=False,
        )
        
        return env
    
    def run_episode(
        self,
        task_name: str,
        episode_idx: int,
    ) -> Dict[str, Any]:
        """Run single LIBERO episode."""
        # Find task index
        task_names = self.get_tasks()
        task_idx = task_names.index(task_name)
        
        video_frames = [] if self.save_videos else None
        
        # Create environment for this task
        env = self._create_env(task_idx)
        
        try:
            obs = env.reset()
            self.policy.reset()
            
            total_reward = 0
            success = False
            
            for step in range(self.max_steps):
                # Get RGB observation
                rgb = obs["agentview_image"]  # (128, 128, 3)
                
                # Get proprioception
                proprio = self._get_proprio(obs)
                
                # Get action from policy
                action = self.policy.predict(
                    rgb=rgb,
                    proprio=proprio,
                )
                
                # Ensure action is correct dimension (7D for LIBERO)
                if len(action) != 7:
                    action = action[:7]
                
                # Execute action
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if self.save_videos:
                    video_frames.append(rgb.copy())
                
                # Check success
                if done or info.get("success", False):
                    success = info.get("success", False)
                    break
            
            # Final success check
            if not success:
                success = env._check_success()
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            success = False
            step = 0
            total_reward = 0
            
        finally:
            env.close()
        
        # Save video if requested
        if self.save_videos and video_frames:
            self._save_video(task_name, episode_idx, video_frames)
        
        return {
            "success": success,
            "steps": step + 1,
            "reward": total_reward,
            "video_frames": video_frames if self.save_videos else None,
        }
    
    def _get_proprio(self, obs: Dict) -> np.ndarray:
        """Extract proprioceptive state from observation."""
        # LIBERO proprioception: ee_pos (3) + ee_quat (4) + gripper (1) = 8D
        proprio_parts = []
        
        if "robot0_eef_pos" in obs:
            proprio_parts.append(obs["robot0_eef_pos"])
        if "robot0_eef_quat" in obs:
            proprio_parts.append(obs["robot0_eef_quat"])
        if "robot0_gripper_qpos" in obs:
            # Take mean of two finger positions for single gripper value
            gripper = np.array([np.mean(obs["robot0_gripper_qpos"])])
            proprio_parts.append(gripper)
        
        if proprio_parts:
            return np.concatenate(proprio_parts)
        return np.zeros(8)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LIBERO Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config yaml (optional)")
    parser.add_argument("--suite", type=str, default="libero_10",
                        choices=LIBERO_SUITES,
                        help="LIBERO task suite")
    parser.add_argument("--n_rollouts", type=int, default=50,
                        help="Number of rollouts per task")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--output_dir", type=str, default="./eval_results/libero",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    evaluator = LIBEROEvaluator(
        task_suite=args.suite,
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        device=args.device,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        seed=args.seed,
        log_wandb=not args.no_wandb,
        save_videos=args.save_videos,
        output_dir=args.output_dir,
    )
    
    evaluator.setup_environment()
    metrics = evaluator.evaluate()
    
    return metrics


if __name__ == "__main__":
    main()

