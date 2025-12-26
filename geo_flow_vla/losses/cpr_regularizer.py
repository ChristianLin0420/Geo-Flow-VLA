"""
Conditional Policy Regularization (CPR) for Geo-Flow VLA.

Implements adversarial regularization for the policy to ensure
generated trajectories are geometrically plausible.

The policy is trained to:
    1. Match expert actions (Flow Matching loss)
    2. Fool the discriminator (CPR loss)

Total Policy Loss:
    L_policy = L_CFM - λ * log(D(s, z_policy))

The discriminator is trained to distinguish:
    - Real: (state, goal) pairs from dataset
    - Fake: (state, goal) pairs where goal comes from policy's backward network

References:
    - GAIL: Generative Adversarial Imitation Learning
    - Adversarial training for policy regularization
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_cpr_loss(
    discriminator: nn.Module,
    state: Tensor,
    policy_goal: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute CPR loss for policy training.
    
    The policy wants to maximize D(s, z_policy), so we minimize -log(D).
    
    Args:
        discriminator: CPRDiscriminator instance
        state: State embeddings (B, state_dim)
        policy_goal: Goals generated/used by policy (B, latent_dim)
        eps: Small constant for numerical stability
        
    Returns:
        CPR loss scalar
    """
    # Get discriminator probability
    logit = discriminator(state, policy_goal)
    prob = torch.sigmoid(logit)
    
    # Policy wants to maximize log(D(s, z))
    # Equivalent to minimizing -log(D(s, z))
    loss = -torch.log(prob + eps).mean()
    
    return loss


class CPRRegularizer(nn.Module):
    """
    CPR Regularizer for policy training.
    
    Provides:
    - Generator (policy) loss computation
    - Discriminator loss computation
    - Scheduled lambda warmup
    - Gradient penalty for stable training
    """

    def __init__(
        self,
        lambda_start: float = 0.0,
        lambda_end: float = 0.1,
        warmup_steps: int = 10000,
        use_gradient_penalty: bool = True,
        gradient_penalty_weight: float = 10.0,
    ) -> None:
        """
        Args:
            lambda_start: Initial CPR weight
            lambda_end: Final CPR weight
            warmup_steps: Steps to linearly increase lambda
            use_gradient_penalty: Use WGAN-GP style gradient penalty
            gradient_penalty_weight: Weight for gradient penalty
        """
        super().__init__()
        
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.warmup_steps = warmup_steps
        self.use_gradient_penalty = use_gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight
        
        # Track training step for lambda scheduling
        self.register_buffer("step", torch.tensor(0))

    def get_lambda(self) -> float:
        """Get current lambda value based on training step."""
        progress = min(1.0, self.step.item() / max(1, self.warmup_steps))
        return self.lambda_start + progress * (self.lambda_end - self.lambda_start)

    def step_scheduler(self) -> None:
        """Increment training step."""
        self.step += 1

    def policy_loss(
        self,
        discriminator: nn.Module,
        state: Tensor,
        policy_goal: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute policy (generator) loss.
        
        The policy wants to fool the discriminator.
        
        Args:
            discriminator: Trained discriminator
            state: State embeddings (B, state_dim)
            policy_goal: Goals from policy's backward network (B, latent_dim)
            
        Returns:
            Dictionary with CPR loss and lambda
        """
        lam = self.get_lambda()
        
        if lam == 0:
            return {
                "cpr_loss": torch.tensor(0.0, device=state.device),
                "cpr_lambda": torch.tensor(lam, device=state.device),
            }
        
        # Get discriminator logit
        logit = discriminator(state, policy_goal)
        
        # Non-saturating GAN loss: -log(D(s, z))
        # More stable than log(1-D) for generator
        cpr_loss = F.softplus(-logit).mean()
        
        return {
            "cpr_loss": lam * cpr_loss,
            "cpr_loss_unweighted": cpr_loss,
            "cpr_lambda": torch.tensor(lam, device=state.device),
            "discriminator_prob": torch.sigmoid(logit).mean(),
        }

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real_state: Tensor,
        real_goal: Tensor,
        fake_state: Tensor,
        fake_goal: Tensor,
        label_smoothing: float = 0.1,
    ) -> Dict[str, Tensor]:
        """
        Compute discriminator loss.
        
        Trains discriminator to distinguish real from fake pairs.
        
        Args:
            discriminator: CPRDiscriminator instance
            real_state: States from dataset (B, state_dim)
            real_goal: Goals from dataset (B, latent_dim)
            fake_state: States for policy goals (B, state_dim)
            fake_goal: Goals generated by policy (B, latent_dim)
            label_smoothing: Label smoothing for stability
            
        Returns:
            Dictionary with discriminator loss and metrics
        """
        # Real loss
        real_logit = discriminator(real_state, real_goal)
        real_label = torch.ones_like(real_logit) * (1 - label_smoothing)
        real_loss = F.binary_cross_entropy_with_logits(real_logit, real_label)
        
        # Fake loss
        fake_logit = discriminator(fake_state.detach(), fake_goal.detach())
        fake_label = torch.zeros_like(fake_logit) + label_smoothing
        fake_loss = F.binary_cross_entropy_with_logits(fake_logit, fake_label)
        
        total_loss = (real_loss + fake_loss) / 2
        
        result = {
            "discriminator_loss": total_loss,
            "discriminator_loss_real": real_loss,
            "discriminator_loss_fake": fake_loss,
        }
        
        # Gradient penalty
        if self.use_gradient_penalty:
            gp = self._gradient_penalty(
                discriminator, real_state, real_goal, fake_state, fake_goal
            )
            result["gradient_penalty"] = gp
            result["discriminator_loss"] = total_loss + self.gradient_penalty_weight * gp
        
        # Accuracy metrics
        with torch.no_grad():
            result["real_accuracy"] = (real_logit > 0).float().mean()
            result["fake_accuracy"] = (fake_logit < 0).float().mean()
        
        return result

    def _gradient_penalty(
        self,
        discriminator: nn.Module,
        real_state: Tensor,
        real_goal: Tensor,
        fake_state: Tensor,
        fake_goal: Tensor,
    ) -> Tensor:
        """Compute gradient penalty for WGAN-GP training."""
        B = real_state.shape[0]
        device = real_state.device
        
        # Random interpolation
        alpha = torch.rand(B, 1, device=device)
        
        interp_state = alpha * real_state + (1 - alpha) * fake_state
        interp_goal = alpha * real_goal + (1 - alpha) * fake_goal
        interp_state.requires_grad_(True)
        interp_goal.requires_grad_(True)
        
        # Forward pass
        logit = discriminator(interp_state, interp_goal)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=logit,
            inputs=[interp_state, interp_goal],
            grad_outputs=torch.ones_like(logit),
            create_graph=True,
            retain_graph=True,
        )
        
        # Combine and compute penalty
        grad_cat = torch.cat([g.view(B, -1) for g in gradients], dim=-1)
        grad_norm = grad_cat.norm(2, dim=-1)
        penalty = ((grad_norm - 1) ** 2).mean()
        
        return penalty

    def combined_policy_loss(
        self,
        cfm_loss: Tensor,
        discriminator: nn.Module,
        state: Tensor,
        policy_goal: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute combined policy loss: CFM + CPR.
        
        L_policy = L_CFM - λ * log(D(s, z_policy))
        
        Note the minus sign: we want to maximize D(s, z),
        which means minimizing -log(D(s, z)).
        
        Args:
            cfm_loss: Flow matching loss
            discriminator: Trained discriminator
            state: State embeddings
            policy_goal: Policy's goal embeddings
            
        Returns:
            Dictionary with total loss and components
        """
        cpr_result = self.policy_loss(discriminator, state, policy_goal)
        
        total_loss = cfm_loss + cpr_result["cpr_loss"]
        
        return {
            "total_loss": total_loss,
            "cfm_loss": cfm_loss,
            **cpr_result,
        }


class AdversarialTrainer:
    """
    Helper class for adversarial training of policy and discriminator.
    
    Handles the alternating optimization scheme.
    """

    def __init__(
        self,
        policy: nn.Module,
        discriminator: nn.Module,
        world_model: nn.Module,
        cpr_regularizer: CPRRegularizer,
        policy_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_steps: int = 1,
    ) -> None:
        """
        Args:
            policy: DiffusionPolicy instance
            discriminator: CPRDiscriminator instance
            world_model: FBWorldModel instance (for getting goal embeddings)
            cpr_regularizer: CPRRegularizer instance
            policy_optimizer: Optimizer for policy
            discriminator_optimizer: Optimizer for discriminator
            discriminator_steps: Number of discriminator updates per policy update
        """
        self.policy = policy
        self.discriminator = discriminator
        self.world_model = world_model
        self.cpr = cpr_regularizer
        self.policy_opt = policy_optimizer
        self.disc_opt = discriminator_optimizer
        self.disc_steps = discriminator_steps

    def train_step(
        self,
        real_state: Tensor,
        real_future_state: Tensor,
        real_actions: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Perform one training step.
        
        Args:
            real_state: Current states from dataset
            real_future_state: Future states from dataset  
            real_actions: Ground truth actions
            
        Returns:
            Dictionary with all losses and metrics
        """
        # Get real goals from future states
        with torch.no_grad():
            real_goal = self.world_model.encode_goal(real_future_state)
        
        # Generate fake goals (policy would use these)
        with torch.no_grad():
            fake_goal = self.world_model.encode_goal(real_state)  # Self-goal
        
        metrics = {}
        
        # Train discriminator
        for _ in range(self.disc_steps):
            self.disc_opt.zero_grad()
            
            disc_result = self.cpr.discriminator_loss(
                self.discriminator,
                real_state, real_goal,
                real_state, fake_goal,  # Same state, different goal
            )
            
            disc_result["discriminator_loss"].backward()
            self.disc_opt.step()
        
        metrics.update({f"disc_{k}": v for k, v in disc_result.items()})
        
        # Train policy
        self.policy_opt.zero_grad()
        
        # CFM loss
        cfm_loss = self.policy.compute_loss(
            real_actions, real_state, real_goal
        )
        
        # Combined loss with CPR
        policy_result = self.cpr.combined_policy_loss(
            cfm_loss, self.discriminator, real_state, fake_goal
        )
        
        policy_result["total_loss"].backward()
        self.policy_opt.step()
        
        metrics.update({f"policy_{k}": v for k, v in policy_result.items()})
        
        # Step CPR scheduler
        self.cpr.step_scheduler()
        
        return metrics

