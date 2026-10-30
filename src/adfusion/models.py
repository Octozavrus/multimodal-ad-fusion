"""3D CNN encoder and seven sMRI + FDG-PET model configurations.

All models share one lightweight encoder (four conv-BN-ReLU-pool blocks,
global average pooling, linear projection). They differ only in where the
two modalities are combined:

    mri, pet        single-modality baselines
    early           two-channel input (requires co-registered volumes)
    late            average of two per-modality decision logits
    intermediate    concatenation of the two feature vectors
    hybrid          bidirectional cross-attention plus per-modality heads,
                    combined by learned softmax weights
    hybrid_noattn   ablation of ``hybrid`` without the cross-attention block

Single-encoder models can be widened (``capacity_matched=True``) so that their
parameter count is close to the two-encoder models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MODEL_NAMES = ["mri", "pet", "early", "late", "intermediate", "hybrid", "hybrid_noattn"]
MATCH_W = 1.41  # about sqrt(2): doubles conv parameters of a one-encoder model


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
    )


class Encoder3D(nn.Module):
    """3D CNN encoder mapping a volume to a ``feat_dim`` feature vector."""

    def __init__(self, in_ch=1, base=(16, 32, 64, 128), feat_dim=128, width_mult=1.0):
        super().__init__()
        widths = [max(4, int(round(w * width_mult))) for w in base]
        blocks, c = [], in_ch
        for w in widths:
            blocks.append(conv_block(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(widths[-1], feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(x)).flatten(1))


class Head(nn.Module):
    """Two-layer classification head returning one logit per sample."""

    def __init__(self, in_dim: int, hidden: int = 64, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SingleModality(nn.Module):
    def __init__(self, which: str = "MRI", feat_dim: int = 128, width_mult: float = 1.0):
        super().__init__()
        if which not in ("MRI", "PET"):
            raise ValueError("which must be 'MRI' or 'PET'")
        self.which = which
        self.enc = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        self.head = Head(feat_dim)

    def forward(self, mri, pet):
        return self.head(self.enc(mri if self.which == "MRI" else pet))


class EarlyFusion(nn.Module):
    """Two-channel input; meaningful only for co-registered volumes."""

    def __init__(self, feat_dim: int = 128, width_mult: float = 1.0):
        super().__init__()
        self.enc = Encoder3D(2, feat_dim=feat_dim, width_mult=width_mult)
        self.head = Head(feat_dim)

    def forward(self, mri, pet):
        return self.head(self.enc(torch.cat([mri, pet], dim=1)))


class LateFusion(nn.Module):
    def __init__(self, feat_dim: int = 128, width_mult: float = 1.0):
        super().__init__()
        self.enc_mri = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        self.enc_pet = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        self.head_mri, self.head_pet = Head(feat_dim), Head(feat_dim)

    def forward(self, mri, pet):
        return 0.5 * (self.head_mri(self.enc_mri(mri)) + self.head_pet(self.enc_pet(pet)))


class IntermediateFusion(nn.Module):
    def __init__(self, feat_dim: int = 128, width_mult: float = 1.0):
        super().__init__()
        self.enc_mri = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        self.enc_pet = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        self.head = Head(feat_dim * 2)

    def forward(self, mri, pet):
        return self.head(torch.cat([self.enc_mri(mri), self.enc_pet(pet)], dim=1))


class CrossAttentionBlock(nn.Module):
    """Queries from one modality, keys and values from the other."""

    def __init__(self, dim: int, n_heads: int = 4, p: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=p, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        out, w = self.attn(q.unsqueeze(1), kv.unsqueeze(1), kv.unsqueeze(1), need_weights=True)
        return self.norm(q + out.squeeze(1)), w


class HybridFusion(nn.Module):
    """Feature-level cross-attention combined with decision-level heads.

    The output logit is ``softmax(w) . [fused, mri, pet]`` where ``w`` is a
    learned 3-vector, so the contribution of each branch can be inspected.
    """

    def __init__(self, feat_dim=128, n_heads=4, p=0.3, width_mult=1.0, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.enc_mri = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        self.enc_pet = Encoder3D(1, feat_dim=feat_dim, width_mult=width_mult)
        if use_attn:
            self.x_m2p = CrossAttentionBlock(feat_dim, n_heads, p)  # MRI queries PET
            self.x_p2m = CrossAttentionBlock(feat_dim, n_heads, p)  # PET queries MRI
        self.fused_head = Head(feat_dim * 2)
        self.head_mri, self.head_pet = Head(feat_dim), Head(feat_dim)
        self.w = nn.Parameter(torch.ones(3))

    def branch_weights(self) -> torch.Tensor:
        """Softmax-normalised weights of the fused, MRI and PET branches."""
        return torch.softmax(self.w.detach(), dim=0)

    def forward(self, mri, pet, return_attn: bool = False):
        fm, fp = self.enc_mri(mri), self.enc_pet(pet)
        if self.use_attn:
            fm_x, w1 = self.x_m2p(fm, fp)
            fp_x, w2 = self.x_p2m(fp, fm)
        else:
            fm_x, fp_x, w1, w2 = fm, fp, None, None
        l_f = self.fused_head(torch.cat([fm_x, fp_x], dim=1))
        l_m, l_p = self.head_mri(fm), self.head_pet(fp)
        w = torch.softmax(self.w, dim=0)
        out = w[0] * l_f + w[1] * l_m + w[2] * l_p
        if return_attn:
            return out, (w1, w2, w.detach())
        return out


def build_model(name: str, feat_dim: int = 128, capacity_matched: bool = False) -> nn.Module:
    """Construct a model by name (see ``MODEL_NAMES``)."""
    n = name.lower()
    wm = MATCH_W if capacity_matched else 1.0
    if n == "mri":
        return SingleModality("MRI", feat_dim, wm)
    if n == "pet":
        return SingleModality("PET", feat_dim, wm)
    if n == "early":
        return EarlyFusion(feat_dim, wm)
    if n == "late":
        return LateFusion(feat_dim)
    if n == "intermediate":
        return IntermediateFusion(feat_dim)
    if n == "hybrid":
        return HybridFusion(feat_dim, use_attn=True)
    if n == "hybrid_noattn":
        return HybridFusion(feat_dim, use_attn=False)
    raise ValueError(f"unknown model '{name}'; choose from {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
