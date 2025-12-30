"""
RLBench-18 Benchmark Evaluation.

Evaluates on 18 RLBench tasks using CoppeliaSim (headless).
Requires: pyrep, rlbench, CoppeliaSim installation.

Usage:
    python -m geo_flow_vla.eval.eval_rlbench \
        --checkpoint ./checkpoints/rlbench/all \
        --n_rollouts 25 \
        --tasks reach_target push_buttons
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# 18 tasks from RLBench benchmark
RLBENCH_18_TASKS = [
    "reach_target",
    "take_lid_off_saucepan",
    "put_item_in_drawer",
    "place_wine_at_rack_location",
    "pick_up_cup",
    "stack_wine",
    "place_cups",
    "put_knife_on_chopping_board",
    "take_umbrella_out_of_umbrella_stand",
    "push_buttons",
    "pick_and_lift",
    "stack_blocks",
    "sweep_to_dustpan_of_size",
    "light_bulb_in",
    "put_groceries_in_cupboard",
    "close_jar",
    "slide_block_to_color_target",
    "meat_off_grill",
]

# Task categories
RLBENCH_TASK_CATEGORIES = {
    "easy": ["reach_target", "pick_up_cup", "push_buttons", "pick_and_lift"],
    "medium": [
        "take_lid_off_saucepan", "put_item_in_drawer", "stack_wine",
        "put_knife_on_chopping_board", "stack_blocks", "close_jar",
        "slide_block_to_color_target", "meat_off_grill",
    ],
    "hard": [
        "place_wine_at_rack_location", "place_cups", 
        "take_umbrella_out_of_umbrella_stand", "sweep_to_dustpan_of_size",
        "light_bulb_in", "put_groceries_in_cupboard",
    ],
}


class RLBenchEvaluator(BaseEvaluator):
    """
    RLBench benchmark evaluator.
    
    Runs policy rollouts in CoppeliaSim headless mode.
    Supports 18 standard RLBench tasks.
    """
    
    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        task_category: Optional[str] = None,
        headless: bool = True,
        camera: str = "front_rgb",
        **kwargs,
    ):
        """
        Args:
            tasks: List of task names (None = all 18)
            task_category: Task category (easy, medium, hard, all)
            headless: Run CoppeliaSim without GUI
            camera: Camera to use (front_rgb, left_shoulder_rgb, etc.)
            **kwargs: Passed to BaseEvaluator
        """
        # Determine tasks to evaluate
        if tasks is not None:
            self.tasks = tasks
        elif task_category is not None:
            if task_category == "all":
                self.tasks = RLBENCH_18_TASKS
            elif task_category in RLBENCH_TASK_CATEGORIES:
                self.tasks = RLBENCH_TASK_CATEGORIES[task_category]
            else:
                raise ValueError(f"Unknown category: {task_category}")
        else:
            self.tasks = RLBENCH_18_TASKS
        
        self.headless = headless
        self.camera = camera
        self.env = None
        
        super().__init__(**kwargs)
    
    @property
    def benchmark_name(self) -> str:
        return "rlbench"
    
    def setup_environment(self, **kwargs) -> None:
        """Initialize RLBench environment."""
        try:
            from rlbench.environment import Environment
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.observation_config import ObservationConfig
            
            # Configure observations
            obs_config = ObservationConfig()
            obs_config.set_all(True)  # Enable all cameras
            
            # Configure action mode: 7DoF EE pose + gripper
            action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=Discrete(),
            )
            
            # Create environment
            self.env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=self.headless,
            )
            self.env.launch()
            
            logger.info(f"✓ RLBench environment launched (headless={self.headless})")
            logger.info(f"  Tasks: {len(self.tasks)}")
            print(f"✓ RLBench environment launched (headless={self.headless})")
            print(f"  Tasks: {len(self.tasks)}")
            
        except ImportError as e:
            raise ImportError(
                f"RLBench not installed: {e}\n"
                "Install with:\n"
                "1. Install CoppeliaSim (https://www.coppeliarobotics.com/)\n"
                "2. export COPPELIASIM_ROOT=/path/to/CoppeliaSim\n"
                "3. pip install pyrep @ git+https://github.com/stepjam/PyRep.git\n"
                "4. pip install rlbench @ git+https://github.com/stepjam/RLBench.git"
            )
    
    def get_tasks(self) -> List[str]:
        return self.tasks
    
    def _get_task_class(self, task_name: str):
        """Get RLBench task class from name."""
        from rlbench.tasks import get_task
        
        # Convert snake_case to CamelCase
        class_name = "".join(word.title() for word in task_name.split("_"))
        
        try:
            return get_task(class_name)
        except Exception:
            raise ValueError(f"Unknown task: {task_name} (tried class: {class_name})")
    
    def run_episode(
        self,
        task_name: str,
        episode_idx: int,
    ) -> Dict[str, Any]:
        """Run single RLBench episode."""
        video_frames = [] if self.save_videos else None
        
        # Get task
        task_class = self._get_task_class(task_name)
        task = self.env.get_task(task_class)
        
        try:
            # Reset task with random variation
            descriptions, obs = task.reset()
            self.policy.reset()
            
            total_reward = 0
            success = False
            
            for step in range(self.max_steps):
                # Get RGB from specified camera
                rgb = getattr(obs, self.camera, None)
                if rgb is None:
                    rgb = obs.front_rgb  # Fallback
                
                # Get proprioception: gripper_pose (7) + gripper_open (1) = 8D
                proprio = np.concatenate([
                    obs.gripper_pose,  # 7D: pos (3) + quat (4)
                    [float(obs.gripper_open)],  # 1D
                ])
                
                # Get action from policy
                action = self.policy.predict(
                    rgb=rgb,
                    instruction=descriptions[0] if descriptions else None,
                    proprio=proprio,
                )
                
                # Ensure action is correct dimension
                # RLBench expects: [x, y, z, qx, qy, qz, qw, gripper]
                if len(action) < 8:
                    # Pad with zeros if needed
                    action = np.concatenate([action, np.zeros(8 - len(action))])
                action = action[:8]
                
                # Execute action
                obs, reward, terminate = task.step(action)
                total_reward += reward
                
                if self.save_videos:
                    video_frames.append(rgb.copy())
                
                if terminate:
                    success = reward > 0  # RLBench: reward > 0 means success
                    break
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            success = False
            step = 0
            total_reward = 0
        
        # Save video if requested
        if self.save_videos and video_frames:
            self._save_video(task_name, episode_idx, video_frames)
        
        return {
            "success": success,
            "steps": step + 1,
            "reward": total_reward,
            "video_frames": video_frames if self.save_videos else None,
        }
    
    def __del__(self):
        """Clean up environment."""
        if self.env is not None:
            try:
                self.env.shutdown()
            except Exception:
                pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RLBench Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config yaml (optional)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all 18)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["easy", "medium", "hard", "all"],
                        help="Task category to evaluate")
    parser.add_argument("--n_rollouts", type=int, default=25,
                        help="Number of rollouts per task")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--camera", type=str, default="front_rgb",
                        help="Camera view to use")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_headless", action="store_true",
                        help="Show CoppeliaSim GUI")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--output_dir", type=str, default="./eval_results/rlbench",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    evaluator = RLBenchEvaluator(
        tasks=args.tasks,
        task_category=args.category,
        headless=not args.no_headless,
        camera=args.camera,
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


