"""3D ConvLSTM with peephole connections over volumetric sequences."""
import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTM3D(nn.Module):
    """x: (B, T, C_in, D, H, W) -> hidden states (B, T, filters, D, H, W).

    Peephole weights are per voxel, so the spatial shape d = (B, T, C_in, D, H, W) is fixed at
    construction. Note: the output-gate peephole uses c_{t-1}, not c_t as in Shi et al. (2015).
    """

    def __init__(self, filters: int, kernel_size: int, pad: str | int, d: tuple):
        super().__init__()
        self.nh = filters
        in_ch = d[2]
        self.Wf = nn.Conv3d(in_ch, filters, kernel_size, padding=pad)
        self.Wi = nn.Conv3d(in_ch, filters, kernel_size, padding=pad)
        self.Wc = nn.Conv3d(in_ch, filters, kernel_size, padding=pad)
        self.Wo = nn.Conv3d(in_ch, filters, kernel_size, padding=pad)
        self.Rf = nn.Conv3d(filters, filters, kernel_size, padding=pad)
        self.Ri = nn.Conv3d(filters, filters, kernel_size, padding=pad)
        self.Rc = nn.Conv3d(filters, filters, kernel_size, padding=pad)
        self.Ro = nn.Conv3d(filters, filters, kernel_size, padding=pad)
        self.Pf = nn.Parameter(torch.randn(1, filters, d[3], d[4], d[5]) * 0.05)
        self.Pi = nn.Parameter(torch.randn(1, filters, d[3], d[4], d[5]) * 0.05)
        self.Po = nn.Parameter(torch.randn(1, filters, d[3], d[4], d[5]) * 0.05)

    def lstm_cell(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        f_gate = torch.sigmoid(self.Wf(x) + self.Rf(h) + self.Pf * c)
        i_gate = torch.sigmoid(self.Wi(x) + self.Ri(h) + self.Pi * c)
        c_gate = torch.tanh(self.Wc(x) + self.Rc(h))
        o_gate = torch.sigmoid(self.Wo(x) + self.Ro(h) + self.Po * c)
        c_next = c * f_gate + c_gate * i_gate
        return torch.tanh(c_next) * o_gate, c_next

    def forward(self, x: Tensor) -> Tensor:
        B, T, _, D, H, W = x.shape
        h = x.new_zeros(B, self.nh, D, H, W)
        c = x.new_zeros(B, self.nh, D, H, W)
        outputs = []
        for t in range(T):
            h, c = self.lstm_cell(x[:, t], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)
