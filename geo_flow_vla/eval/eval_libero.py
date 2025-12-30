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
import os

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
        self.benchmark = None
        self.task_descriptions = None
        
        super().__init__(**kwargs)
        
    @property
    def benchmark_name(self) -> str:
        return f"libero_{self.task_suite}"
    
    def setup_environment(self, **kwargs) -> None:
        """Initialize LIBERO environments using official benchmark API."""
        try:
            from libero.libero.benchmark import get_benchmark
            from libero.libero import get_libero_path
            
            # Use official LIBERO benchmark API
            benchmark_dict = get_benchmark(self.task_suite)
            self.benchmark = benchmark_dict()
            
            # Get task language descriptions from task objects
            self.task_descriptions = []
            for i in range(self.benchmark.n_tasks):
                task = self.benchmark.get_task(i)
                self.task_descriptions.append(task.language)
            
            n_tasks = self.benchmark.n_tasks
            logger.info(f"✓ LIBERO {self.task_suite}: {n_tasks} tasks")
            print(f"✓ LIBERO {self.task_suite}: {n_tasks} tasks")
            
            # Print task names
            for i, name in enumerate(self.benchmark.get_task_names()):
                logger.debug(f"  Task {i}: {name}")
            
        except ImportError as e:
            raise ImportError(
                f"LIBERO not installed: {e}\n"
                "Install with:\n"
                "  cd /tmp && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
                "  touch /tmp/LIBERO/libero/__init__.py\n"
                "  pip install -e /tmp/LIBERO\n"
                "  pip install robosuite==1.4.1"
            )
    
    def get_tasks(self) -> List[str]:
        """Return task names in suite."""
        if self.benchmark is None:
            self.setup_environment()
        return self.benchmark.get_task_names()
    
    def _create_env(self, task_idx: int):
        """Create environment for a specific task using LIBERO benchmark API."""
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import get_libero_path
        
        # Get task from benchmark
        task = self.benchmark.get_task(task_idx)
        
        # Get BDDL file path - need full path
        bddl_files_path = get_libero_path("bddl_files")
        bddl_file = os.path.join(bddl_files_path, task.problem_folder, task.bddl_file)
        
        # Get init states
        init_states = self.benchmark.get_task_init_states(task_idx)
        
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
        
        # Store init states for resetting
        env._init_states = init_states
        
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
            # Reset with init state if available
            init_states = env._init_states if hasattr(env, '_init_states') else None
            if init_states is not None and len(init_states) > 0:
                # Pick an init state based on episode index
                init_state_idx = episode_idx % len(init_states)
                obs = env.reset()
                env.set_init_state(init_states[init_state_idx])
                # Step with zero action to get updated observation
                obs, _, _, _ = env.step(np.zeros(7))
            else:
                obs = env.reset()
            
            self.policy.reset()
            
            total_reward = 0
            success = False
            step = 0
            
            for step in range(self.max_steps):
                # Get RGB observation
                rgb = obs["agentview_image"]  # (128, 128, 3)
                
                # Get proprioception
                proprio = self._get_proprio(obs)
                
                # Get language instruction if available
                instruction = None
                if self.task_descriptions is not None and task_idx < len(self.task_descriptions):
                    instruction = self.task_descriptions[task_idx]
                
                # Get action from policy
                action = self.policy.predict(
                    rgb=rgb,
                    proprio=proprio,
                    instruction=instruction,
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
                success = env.check_success()
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
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

