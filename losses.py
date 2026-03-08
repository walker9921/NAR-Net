import torch.nn as nn
import torch.nn.functional as F
import torch
from contextlib import nullcontext
from einops import rearrange


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, patch_size=0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        self.ps = patch_size

    def forward(self, pred, target):
        device_type = pred.device.type if pred.is_cuda or pred.device.type == 'cpu' else 'cpu'
        autocast_ctx = torch.autocast(device_type=device_type, enabled=False) if torch.is_autocast_enabled() else nullcontext()

        with autocast_ctx:
            pred_fp32 = pred.float()
            target_fp32 = target.float()

            if self.ps > 0:
                B, C, H, W = pred_fp32.size()

                grid_height, grid_width = H // self.ps, W // self.ps
                pred_patch = rearrange(
                    pred_fp32, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw",
                    gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps)

                target_patch = rearrange(
                    target_fp32, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw",
                    gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps)

                pred_fft = torch.fft.rfft2(pred_patch, dim=(-2, -1))
                target_fft = torch.fft.rfft2(target_patch, dim=(-2, -1))

                pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
                target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

            else:
                pred_fft = torch.fft.rfft2(pred_fp32, dim=(-2, -1))
                target_fft = torch.fft.rfft2(target_fp32, dim=(-2, -1))

                pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
                target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

            loss = self.loss_weight * self.criterion(pred_fft, target_fft)

        return loss


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-5

    def forward(self, X, Y):
        device_type = X.device.type if X.is_cuda or X.device.type == 'cpu' else 'cpu'
        autocast_ctx = torch.autocast(device_type=device_type, enabled=False) if torch.is_autocast_enabled() else nullcontext()
        
        with autocast_ctx:
            X_fp32 = X.float()
            Y_fp32 = Y.float()
            diff = torch.add(X_fp32, -Y_fp32)
            loss = torch.mean(torch.sqrt( diff * diff + self.eps))
        return loss


class HybridWPTLoss(nn.Module):
    """Weighted proximal trajectory loss following ProxUnroll training."""

    def __init__(self, stages: int, emphasis: str = 'uniform', eps: float = 1e-8):
        super().__init__()
        self.stages = max(1, int(stages))
        self.eps = eps
        weights = self._generate_weights(self.stages, emphasis)
        self.stage_weights: torch.Tensor
        self.register_buffer('stage_weights', weights)
        # Mirror train_proxunroll.py: use MSE and take square-root per stage
        self.criterion = nn.MSELoss(reduction='mean')

    @staticmethod
    def _generate_weights(stages: int, emphasis: str) -> torch.Tensor:
        if emphasis == 'last' and stages >= 1:
            weights = torch.full((stages,), 0.01, dtype=torch.float32)
            weights[-1] = 0.95
            return weights
        base = torch.ones(stages, dtype=torch.float32)
        return base / base.sum().clamp_min(1e-8)

    def forward(self, Uk_student, ProxIdeal_target):
        if not Uk_student or not ProxIdeal_target:
            device = Uk_student[0].device if Uk_student else (ProxIdeal_target[0].device if ProxIdeal_target else torch.device('cpu'))
            return torch.tensor(0.0, device=device)

        device = Uk_student[0].device
        K = min(len(Uk_student), len(ProxIdeal_target), self.stages)
        weights = self.stage_weights[:K].to(device)

        loss_total = torch.tensor(0.0, device=device)
        for idx in range(K):
            # Cast to float for stability
            input_s = Uk_student[idx].float()
            target_s = ProxIdeal_target[idx].float()
            mse = self.criterion(input_s, target_s)
            stage_loss = torch.sqrt(mse.clamp_min(self.eps))
            loss_total = loss_total + weights[idx] * stage_loss

        return loss_total


class DynamicJointOrthogonalityLoss(nn.Module):
    """
    Dynamic Joint Orthogonality Loss (Fusion of Frame Potential and Subspace Mutual Exclusion).
    Minimizes the sum of squared inner products of all pairs of measurement vectors.
    L = sum_{k != l} (<a_k, a_l>)^2
    where a_k = psi_{i,u} (x) phi_i
    """
    def __init__(self):
        super(DynamicJointOrthogonalityLoss, self).__init__()

    def forward(self, phi, psi):
        """
        Args:
            phi: (m, H)
            psi: (m, n, W)
        """
        phi = phi.float()
        psi = psi.float()
        if phi.dim() == 3 and phi.shape[1] == 1:
            phi = phi.squeeze(1)
        m, H = phi.shape
        m2, n, W = psi.shape
        assert m == m2, f"phi and psi must have same number of blocks, got {m} and {m2}"
        
        # Normalize vectors to ensure we are optimizing correlation/angles
        phi = F.normalize(phi, p=2, dim=1)
        psi = F.normalize(psi, p=2, dim=2)
        
        # Precompute Gram matrix of phi (m x m)
        g_phi = torch.mm(phi, phi.t())
        g_phi_sq = g_phi.pow(2)
        
        loss = 0.0
        
        # Compute loss using block decomposition:
        # L_total = sum_{i,j} ( <phi_i, phi_j>^2 * ||Psi_i Psi_j^T||_F^2 )
        # We iterate over i to avoid creating (m, m, n, n) tensor if memory is constrained.
        
        for i in range(m):
            # Compute cross-Gram matrix between block i and all blocks
            # psi[i]: (n, W)
            # psi: (m, n, W)
            # d_i: (m, n, n) where d_i[j, u, v] = <psi_{i,u}, psi_{j,v}>
            d_i = torch.einsum('uw, mjw -> muj', psi[i], psi)
            
            # b_i[j] = ||Psi_i Psi_j^T||_F^2 = sum_{u,v} d_i[j, u, v]^2
            b_i = d_i.pow(2).sum(dim=(1, 2))
            
            # Accumulate weighted by phi correlation
            loss += (g_phi_sq[i] * b_i).sum()
            
        # Subtract diagonal elements (k == l)
        # Since vectors are normalized, <a_k, a_k>^2 = 1.
        # Total number of measurements M_total = m * n
        total_measurements = m * n
        loss = loss - total_measurements
        
        return loss
