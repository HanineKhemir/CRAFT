"""
asam.py — Adaptive Sharpness-Aware Minimization (ASAM) optimizer.

ASAM improves on SAM by scaling the perturbation radius per-parameter
according to the parameter's magnitude, making it more adaptive and
better suited to mixed-precision / heterogeneous architectures like AST.

Reference: Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization
for Scale-Invariant Learning of Deep Neural Networks" (ICML 2021).
"""
from __future__ import annotations

import torch


class ASAM:
    """
    Adaptive SAM optimizer wrapper.

    Wraps any base optimizer (e.g. AdamW) with ASAM's two-step update:
      Step 1 (ascent):  perturb weights to find worst-case direction.
      Step 2 (descent): restore weights, compute loss at perturbed point,
                        take gradient step with the base optimizer.

    Usage in training loop:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        asam = ASAM(optimizer, model, rho=0.5, eta=0.01)

        # Forward pass at original weights
        outputs = model(...)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        asam.ascent_step()   # perturb weights

        # Forward pass at perturbed weights
        outputs = model(...)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        asam.descent_step()  # restore + gradient step

    Args:
        optimizer:  Base optimizer (AdamW recommended).
        model:      The model being optimized.
        rho:        Perturbation radius (default 0.5).
        eta:        Small constant for numerical stability (default 0.01).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        rho: float = 0.5,
        eta: float = 0.01,
    ):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self._saved_weights: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def ascent_step(self) -> None:
        """
        Step 1: Compute adaptive perturbation and move weights to worst-case point.
        Saves current weights so they can be restored in descent_step.
        """
        grad_norm = self._grad_norm()

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # Save original weight
            self._saved_weights[name] = param.data.clone()
            # Adaptive scale: rho * |w| / ||grad||
            scale = self.rho / (grad_norm + 1e-12)
            e_w = (torch.pow(param, 2) * param.grad) * scale
            # Add perturbation
            param.add_(e_w)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self) -> None:
        """
        Step 2: Restore original weights, then take the gradient step.
        """
        for name, param in self.model.named_parameters():
            if name in self._saved_weights:
                param.data = self._saved_weights[name]

        self._saved_weights.clear()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        """Compute the L2 norm of all gradients."""
        grads = [
            p.grad.norm(p=2)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not grads:
            return torch.tensor(0.0)
        return torch.stack(grads).norm(p=2)

    # Expose base optimizer interface
    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def build_optimizer(model: torch.nn.Module, cfg) -> tuple:
    """
    Build the optimizer (and ASAM wrapper if configured).

    Returns:
        optimizer: The base optimizer (always AdamW).
        asam:      ASAM wrapper, or None if optimizer.type != 'asam'/'sam'.
    """
    opt_cfg = cfg.optimizer
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.base_lr,
        weight_decay=opt_cfg.weight_decay,
    )

    opt_type = opt_cfg.type.lower()
    if opt_type == "asam":
        asam = ASAM(base_optimizer, model, rho=opt_cfg.asam_rho, eta=opt_cfg.asam_eta)
        return base_optimizer, asam
    elif opt_type == "sam":
        # Simplified SAM: same interface but uniform rho (no adaptive scaling)
        asam = ASAM(base_optimizer, model, rho=opt_cfg.get("asam_rho", 0.05), eta=0.0)
        return base_optimizer, asam
    else:
        return base_optimizer, None