"""Forward pass shared by both CycleGAN variants (they differ only in their components)."""
import torch.nn as nn
from torch import Tensor

from moco.models.outputs import ModelOutputs


def masked_residual(base: Tensor, delta: Tensor) -> Tensor:
    """base + delta, with delta zeroed outside base's brain (base != 0) so background stays exactly 0."""
    return base + delta * (base != 0)


class CycleGANBase(nn.Module):
    """Subclasses set E_c, E_a, G_B, G_A, D_A, D_B and self.residual."""

    residual: bool

    def generator_parameters(self) -> list:
        return (list(self.E_c.parameters()) + list(self.E_a.parameters())
                + list(self.G_B.parameters()) + list(self.G_A.parameters()))

    def discriminator_parameters(self) -> list:
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        content = self.E_c(x)
        a_global, a_spatial = self.E_a(x)
        return content, a_global, a_spatial

    def _decoded(self, base: Tensor, out: Tensor) -> Tensor:
        return masked_residual(base, out) if self.residual else out

    def forward(self, x_a: Tensor, x_b: Tensor) -> ModelOutputs:
        """x_a: (B, T, D, H, W) motion-corrupted, x_b: (B, T, D, H, W) motion-free."""
        c_a, a_global, a_spatial = self.encode(x_a)
        c_b, a_global_b, a_spatial_b = self.encode(x_b)

        x_hat_b = self._decoded(x_a, self.G_B(c_a))
        x_hat_a = self._decoded(x_b, self.G_A(c_b, a_global, a_spatial))
        x_self_a = self._decoded(x_a, self.G_A(c_a, a_global, a_spatial))
        x_self_b = self._decoded(x_b, self.G_B(c_b))

        c_hat_b = self.E_c(x_hat_b)
        c_hat_a, a_hat_global, a_hat_spatial = self.encode(x_hat_a)
        x_cycle_a = self._decoded(x_hat_b, self.G_A(c_hat_b, a_hat_global, a_hat_spatial))
        x_cycle_b = self._decoded(x_hat_a, self.G_B(c_hat_a))

        return ModelOutputs(
            c_a=c_a, c_b=c_b, a_global=a_global, a_spatial=a_spatial,
            a_global_b=a_global_b, a_spatial_b=a_spatial_b,
            x_hat_b=x_hat_b, x_hat_a=x_hat_a, x_self_a=x_self_a, x_self_b=x_self_b,
            c_hat_b=c_hat_b, c_hat_a=c_hat_a, a_hat_global=a_hat_global, a_hat_spatial=a_hat_spatial,
            x_cycle_a=x_cycle_a, x_cycle_b=x_cycle_b,
            score_real_b=self.D_B(x_b), score_fake_b=self.D_B(x_hat_b),
            score_real_a=self.D_A(x_a), score_fake_a=self.D_A(x_hat_a),
        )

    def correct(self, x_a: Tensor) -> Tensor:
        """Inference path E_c -> G_B. x_a: (B, T, D, H, W) -> corrected (B, T, D, H, W)."""
        return self._decoded(x_a, self.G_B(self.E_c(x_a)))

    def count_parameters(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())
        counts = {name: n(getattr(self, name)) for name in ("E_c", "E_a", "G_B", "G_A", "D_B", "D_A")}
        counts["total"] = n(self)
        return counts
