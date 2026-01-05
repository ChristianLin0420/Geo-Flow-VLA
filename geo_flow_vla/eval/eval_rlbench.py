"""
RLBench-18 Benchmark Evaluation.

Evaluates on 18 RLBench tasks using CoppeliaSim (headless).
Requires: pyrep, rlbench, CoppeliaSim installation.

Usage (explicit paths - recommended):
    python -m geo_flow_vla.eval.eval_rlbench \
        --world_model_path ./checkpoints/phase1/world_model.pth \
        --policy_path ./checkpoints/phase2/best.pt \
        --category all \
        --n_rollouts 25

Usage (legacy checkpoint directory):
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

# 18 tasks from RLBench-18-Tasks HuggingFace dataset (MUST match rlbench_dataset.py)
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

# Task categories (matches rlbench_dataset.py and rlbench_config.yaml)
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
            # Use EndEffectorPoseViaPlanning for absolute pose control
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
        import importlib
        
        # Convert snake_case to CamelCase
        class_name = "".join(word.title() for word in task_name.split("_"))
        
        try:
            # Import the task module dynamically
            module = importlib.import_module(f"rlbench.tasks.{task_name}")
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown task: {task_name} (tried class: {class_name}): {e}")
    
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
            
            # Set instruction for language-conditioned policy
            instruction = descriptions[0] if descriptions else f"complete the {task_name} task"
            if hasattr(self.policy, 'set_instruction'):
                self.policy.set_instruction(instruction)
            
            total_reward = 0
            success = False
            step = 0
            consecutive_failures = 0
            max_consecutive_failures = 5  # Allow some path planning failures
            
            for step in range(self.max_steps):
                # Get RGB from specified camera
                rgb = getattr(obs, self.camera, None)
                if rgb is None:
                    rgb = obs.front_rgb  # Fallback
                
                # Capture frame for video BEFORE action (so we see all steps)
                if self.save_videos:
                    video_frames.append(rgb.copy())
                
                # Get proprioception: gripper_pose (7) + gripper_open (1) = 8D
                proprio = np.concatenate([
                    obs.gripper_pose,  # 7D: pos (3) + quat (4)
                    [float(obs.gripper_open)],  # 1D
                ])
                
                # Get action from policy (instruction already set)
                action = self.policy.predict(
                    rgb=rgb,
                    instruction=instruction,
                    proprio=proprio,
                )
                
                # Ensure action is correct dimension
                # RLBench expects: [x, y, z, qx, qy, qz, qw, gripper]
                if len(action) < 8:
                    # Pad with zeros if needed
                    action = np.concatenate([action, np.zeros(8 - len(action))])
                action = action[:8].copy()
                
                # Get current EE pose for reference (used as fallback)
                current_quat = obs.gripper_pose[3:7]
                
                # Normalize quaternion to unit quaternion (CRITICAL - RLBench requires this)
                quat = action[3:7]
                quat_norm = np.linalg.norm(quat)
                if quat_norm > 1e-6:
                    action[3:7] = quat / quat_norm
                else:
                    # Keep current rotation if quaternion is invalid
                    action[3:7] = current_quat
                
                # Clip gripper to valid range
                action[7] = np.clip(action[7], 0.0, 1.0)
                
                # Try to execute action with error recovery
                try:
                    obs, reward, terminate = task.step(action)
                    total_reward += reward
                    consecutive_failures = 0  # Reset on success
                    
                    if terminate:
                        success = reward > 0  # RLBench: reward > 0 means success
                        break
                        
                except Exception as step_error:
                    # Path planning can fail for some target poses
                    consecutive_failures += 1
                    error_msg = str(step_error)
                    
                    # CRITICAL: Clear action buffer on failure!
                    # The buffered actions were predicted assuming this action succeeded.
                    # Since it failed, the robot hasn't moved, so we need fresh predictions.
                    self.policy.reset()
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Step {step}: Too many consecutive failures, ending episode")
                        break
                    
                    # Log but continue - some failures are recoverable
                    if "path could not be found" in error_msg.lower():
                        logger.debug(f"Step {step}: Path planning failed, will re-predict...")
                    elif "quaternion" in error_msg.lower():
                        logger.debug(f"Step {step}: Quaternion error, will re-predict...")
                    else:
                        logger.warning(f"Step {step}: {error_msg[:60]}, will re-predict...")
            
        except Exception as e:
            logger.error(f"Episode failed during reset: {e}")
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
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all 18)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["easy", "medium", "hard", "all"],
                        help="Task category to evaluate")
    parser.add_argument("--n_rollouts", type=int, default=25,
                        help="Number of rollouts per task")
    parser.add_argument("--max_steps", type=int, default=500,
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
    
    evaluator = RLBenchEvaluator(
        tasks=args.tasks,
        task_category=args.category,
        headless=not args.no_headless,
        camera=args.camera,
        world_model_path=args.world_model_path,
        policy_path=args.policy_path,
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        device=args.device,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
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


