"""Latent AMOC model with a four-surface public input contract.

The checkpoint retains an inactive fifth (wind) slot. It is supplied internally
as zero so saved weights load without needing a wind dataset or exposing wind
as a predictor. The public predictors are the four surface fields (SST, SSS,
SSH, OBP) plus the virtual-Argo density profiles.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import log
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.utils as pyg_utils

    _PYG_AVAILABLE = True
    _PYG_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    pyg_nn = None
    pyg_utils = None
    _PYG_AVAILABLE = False
    _PYG_IMPORT_ERROR = exc


SigmaPolicy = Literal[
    "all_learnable",
    "days_fixed_spatial_learnable",
    "all_fixed",
]
_SIGMA_DIM_NAMES: Tuple[str, str, str] = ("lon", "lat", "days")      # order of the three assignment coordinates
_SURFACE_SLOT_ORDER: Tuple[str, ...] = ("ws", "sst", "sss", "ssh", "obp")   # fixed slot order; ws is zeroed in the final method
GateMode = Literal["baseline_only", "seasonal"]
ActivationName = Literal["relu", "gelu", "silu"]
NormType = Literal["layernorm", "none"]
SurfaceVariableWeighting = Literal["gate", "none", "learned_scalar"]
GCNNormalization = Literal["rw", "sym", "none"]


class DistanceAssignerModule(nn.Module):
    """Soft assignment of Argo profiles to C compartments via a Gaussian kernel.

    Each compartment k has a learnable centre c_k in (lon, lat, days) and a
    per-dimension width sigma_k. The probability that profile x belongs to
    compartment k is softmax over k of -sum_d (x_d - c_{k,d})^2 / sigma_{k,d}^2.

    Forward input:
      - `(B, P, 3)` or `(P, 3)`, where the last dimension is `[lon, lat, days]`.

    Forward output:
      - `(B, P, C)` or `(P, C)`, containing a probability over compartments.
    """

    def __init__(
        self,
        n_compartments: int,
        centers: Optional[np.ndarray] = None,
        centers_learnable: bool = True,
        sigmas_learnable: bool = True,
        sigma_policy: SigmaPolicy = "all_learnable",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.n_compartments = int(n_compartments)
        self._dtype = dtype
        self.centers_learnable = bool(centers_learnable)
        self.sigma_policy: SigmaPolicy = str(sigma_policy)

        if self.n_compartments <= 0:
            raise ValueError(f"n_compartments must be > 0, got {self.n_compartments}")
        if self.sigma_policy not in {
            "all_learnable",
            "days_fixed_spatial_learnable",
            "all_fixed",
        }:
            raise ValueError(
                "sigma_policy must be 'all_learnable', "
                f"'days_fixed_spatial_learnable', or 'all_fixed', got {self.sigma_policy!r}"
            )

        # Compartment centres (C, 3) in (lon, lat, days). Initialised either from
        # the supplied array (e.g. KMeans) or from a random cube in [-shift, scale-shift].
        if centers is not None:
            centers_t = torch.tensor(centers, dtype=self._dtype)
            if centers_t.shape != (self.n_compartments, 3):
                raise ValueError(
                    f"centers must be ({self.n_compartments}, 3), got {tuple(centers_t.shape)}"
                )
            self.centers = nn.Parameter(centers_t, requires_grad=self.centers_learnable)
        else:
            rand = torch.rand((self.n_compartments, 3), dtype=self._dtype)
            scale = torch.tensor([1.0, 1.0, 2.0], dtype=self._dtype)
            shift = torch.tensor([0.0, 0.0, 1.0], dtype=self._dtype)
            self.centers = nn.Parameter(
                rand * scale - shift,
                requires_grad=self.centers_learnable,
            )

        # Per-compartment, per-dimension Gaussian widths. The policy mask decides
        # which dimensions are optimised; frozen dimensions use _sigma_fixed_values.
        self.sigmas = nn.Parameter(
            torch.ones((1, self.n_compartments, 3), dtype=self._dtype),
            requires_grad=bool(sigmas_learnable),
        )
        trainable_mask = torch.ones((1, self.n_compartments, 3), dtype=self._dtype)
        if self.sigma_policy == "days_fixed_spatial_learnable":
            trainable_mask[..., 2] = 0.0                     # freeze days sigma
        elif self.sigma_policy == "all_fixed":
            trainable_mask.zero_()
        self.register_buffer("_sigma_trainable_mask", trainable_mask)
        self.register_buffer("_sigma_fixed_values", torch.ones((1, self.n_compartments, 3), dtype=self._dtype))

    def sigma_values_raw(self) -> torch.Tensor:
        """Return strictly positive raw sigma values before policy masking."""
        return torch.abs(self.sigmas) + 1e-6

    def sigma_values(self) -> torch.Tensor:
        """Return strictly positive sigma values after the active policy is applied."""
        raw = self.sigma_values_raw()
        # Trainable dims use the current parameter; frozen dims use the fixed value.
        return (
            raw * self._sigma_trainable_mask
            + self._sigma_fixed_values * (1.0 - self._sigma_trainable_mask)
        )

    def sigma_trainable_mask(self) -> torch.Tensor:
        return self._sigma_trainable_mask.detach().clone()

    def sigma_dimension_names(self) -> Tuple[str, str, str]:
        return _SIGMA_DIM_NAMES

    def sigma_trainable_dimension_names(self) -> Tuple[str, ...]:
        names = []
        mask = self._sigma_trainable_mask[0, 0].detach().cpu().numpy()
        for name, flag in zip(_SIGMA_DIM_NAMES, mask):
            if float(flag) > 0.5:
                names.append(name)
        return tuple(names)

    def sigma_fixed_dimension_names(self) -> Tuple[str, ...]:
        names = []
        mask = self._sigma_trainable_mask[0, 0].detach().cpu().numpy()
        for name, flag in zip(_SIGMA_DIM_NAMES, mask):
            if float(flag) < 0.5:
                names.append(name)
        return tuple(names)

    def apply_sigma_init(self, sigma_init: float) -> None:
        """Fill both the trainable sigmas and the fixed-value buffer with sigma_init."""
        sigma_init = float(sigma_init)
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        with torch.no_grad():
            self.sigmas.data.fill_(sigma_init)
            self._sigma_fixed_values.fill_(sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the soft assignment probability of every input position over the C compartments."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.Tensor, got {type(x)}")
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dim=3 for [lon, lat, days], got {x.shape[-1]}")

        x = x.to(dtype=self.centers.dtype, device=self.centers.device)
        sigmas = self.sigma_values()

        # Broadcast centres so the squared-distance tensor matches (..., C, 3).
        if x.ndim == 3:
            centers = self.centers.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            centers = self.centers.unsqueeze(0)
        else:
            raise ValueError(f"x must be 2D or 3D, got shape {tuple(x.shape)}")

        dist = ((x.unsqueeze(-2) - centers) ** 2 / sigmas**2).sum(dim=-1)   # (..., C) Mahalanobis-like squared distance
        return torch.softmax(-dist, dim=-1)                                 # softmax over compartments, sums to 1

    def get_adjacent_matrix(self) -> torch.Tensor:
        """Return a (C, C) soft adjacency matrix built from pairwise distances between compartment centres."""
        return torch.softmax(-torch.cdist(self.centers, self.centers) ** 2, dim=-1)


class BysGNN_GNN_GCN(nn.Module):
    """ChebConv graph block over the compartment graph.

    First layer: ChebConv + ReLU (dimension changes from node_feature_dim to
    conv_hidden_dim, so no residual). Subsequent layers: ChebConv + residual
    sum + LayerNorm + ReLU. An optional node mask zeroes out inactive
    compartments after every layer.
    """

    def __init__(
        self,
        node_feature_dim: int,
        conv_hidden_dim: int = 64,
        conv_num_layers: int = 3,
        cheb_k: int = 3,
        gcn_normalization: GCNNormalization = "rw",
    ):
        super().__init__()

        self.node_feature_dim = int(node_feature_dim)
        self.conv_hidden_dim = int(conv_hidden_dim)
        self.conv_layers_num = int(conv_num_layers)
        self.cheb_k = int(cheb_k)
        self.gcn_normalization: GCNNormalization = str(gcn_normalization)

        if self.node_feature_dim <= 0:
            raise ValueError(f"node_feature_dim must be > 0, got {self.node_feature_dim}")
        if self.conv_hidden_dim <= 0:
            raise ValueError(f"conv_hidden_dim must be > 0, got {self.conv_hidden_dim}")
        if self.conv_layers_num <= 0:
            raise ValueError(f"conv_num_layers must be > 0, got {self.conv_layers_num}")
        if self.cheb_k <= 0:
            raise ValueError(f"cheb_k must be > 0, got {self.cheb_k}")
        if self.gcn_normalization not in {"rw", "sym", "none"}:
            raise ValueError(
                "gcn_normalization must be 'rw', 'sym', or 'none', "
                f"got {self.gcn_normalization!r}"
            )
        if not _PYG_AVAILABLE:
            raise ImportError(
                "blocked_kfold2 model requires torch_geometric in the intended project "
                f"environment, but it is unavailable here: {_PYG_IMPORT_ERROR!r}"
            )

        normalization_arg = None if self.gcn_normalization == "none" else self.gcn_normalization
        # First conv: node_feature_dim -> conv_hidden_dim. Later convs keep conv_hidden_dim.
        self.convs = nn.ModuleList(
            [
                pyg_nn.ChebConv(
                    self.node_feature_dim,
                    self.conv_hidden_dim,
                    K=self.cheb_k,
                    normalization=normalization_arg,
                )
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num - 1)]
        )
        for _ in range(self.conv_layers_num - 1):
            self.convs.append(
                pyg_nn.ChebConv(
                    self.conv_hidden_dim,
                    self.conv_hidden_dim,
                    K=self.cheb_k,
                    normalization=normalization_arg,
                )
            )

    @property
    def uses_torch_geometric(self) -> bool:
        return True

    def forward(
        self,
        X: torch.Tensor,
        adj_mat: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run every graph in the batch through the ChebConv stack."""
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"X must be torch.Tensor, got {type(X)}")
        if not isinstance(adj_mat, torch.Tensor):
            raise TypeError(f"adj_mat must be torch.Tensor, got {type(adj_mat)}")
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (B, C, E), got {tuple(X.shape)}")
        if adj_mat.ndim != 2:
            raise ValueError(f"adj_mat must be 2D (C, C), got {tuple(adj_mat.shape)}")
        if adj_mat.shape != (X.shape[1], X.shape[1]):
            raise ValueError(
                f"adj_mat shape {tuple(adj_mat.shape)} must match compartment axis {X.shape[1]}"
            )

        if node_mask is not None:
            if node_mask.shape != (X.shape[0], X.shape[1], 1):
                raise ValueError(
                    f"node_mask must be (B, C, 1), got {tuple(node_mask.shape)}"
                )
            node_mask = node_mask.to(dtype=X.dtype, device=X.device)

        adj_mat = adj_mat.to(dtype=X.dtype, device=X.device)
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)       # ChebConv consumes a COO edge list
        # Process each graph in the batch independently; the adjacency is shared across the batch.
        outputs = []
        for i in range(X.shape[0]):
            X_i = X[i]
            mask_i = node_mask[i] if node_mask is not None else None

            # First conv: no residual (input and output dims differ).
            X_i = F.relu(self.convs[0](X_i, edge_indices, edge_attrs))
            if mask_i is not None:
                X_i = X_i * mask_i

            # Remaining convs: residual + LayerNorm + ReLU, then re-mask.
            for stack_i in range(1, self.conv_layers_num):
                X_res = X_i
                X_i = self.convs[stack_i](X_i, edge_indices, edge_attrs)
                X_i = F.relu(self.layer_norms[stack_i - 1](X_i + X_res))
                if mask_i is not None:
                    X_i = X_i * mask_i

            outputs.append(X_i)

        return torch.stack(outputs, dim=0)


def _logit_probability(probability: float) -> float:
    probability = float(probability)
    if not (0.0 < probability < 1.0):
        raise ValueError(f"probability must lie strictly inside (0, 1), got {probability}")
    return log(probability / (1.0 - probability))


def _validate_dropout(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{name} must lie in [0, 1), got {value}")
    return value


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"activation must be one of 'relu', 'gelu', 'silu', got {name!r}")


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(int(dim))
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"norm_type must be 'layernorm' or 'none', got {norm_type!r}")


def _scaled_hidden_dim(base_dim: int, multiplier: float, *, name: str) -> int:
    """Scale a base dimension by a positive multiplier and round to at least 1."""
    multiplier = float(multiplier)
    if multiplier <= 0.0:
        raise ValueError(f"{name} must be > 0, got {multiplier}")
    return max(1, int(round(int(base_dim) * multiplier)))


def _make_latent_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    *,
    depth: int = 1,
    activation: ActivationName = "relu",
    norm_type: NormType = "layernorm",
) -> nn.Sequential:
    """Build an MLP: depth hidden blocks (Linear + activation + norm + dropout), then a projection to output_dim."""
    if int(input_dim) <= 0:
        raise ValueError(f"input_dim must be > 0, got {input_dim}")
    if int(hidden_dim) <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
    if int(output_dim) <= 0:
        raise ValueError(f"output_dim must be > 0, got {output_dim}")
    depth = int(depth)
    if depth <= 0:
        raise ValueError(f"depth must be > 0, got {depth}")
    dropout = _validate_dropout("dropout", dropout)
    layers = []
    prev_dim = int(input_dim)
    for _ in range(depth):
        layers.extend(
            [
                nn.Linear(prev_dim, int(hidden_dim)),
                _make_activation(activation),
                _make_norm(norm_type, int(hidden_dim)),
                nn.Dropout(p=dropout),
            ]
        )
        prev_dim = int(hidden_dim)
    layers.extend(
        [
            nn.Linear(prev_dim, int(output_dim)),
            _make_norm(norm_type, int(output_dim)),
            nn.Dropout(p=dropout),
        ]
    )
    return nn.Sequential(*layers)


class SurfaceLatentModule(nn.Module):
    """Variable-aware latent surface encoder.

    For every surface slot (ws, sst, sss, ssh, obp) a dedicated MLP encodes the
    monthly bin vector into a small latent of size z_var_dim. Each slot is then
    scaled by a sigmoid gate (optionally season-dependent). Ablated slots are
    zeroed out. The five slot latents are concatenated and fed to a fusion MLP
    that produces the shared surface_latent of size surface_latent_dim.
    """

    def __init__(
        self,
        *,
        surface_feature_dims: Dict[str, int],
        surface_native_feature_dims: Dict[str, int],
        surface_months_per_sample: int,
        n_season_features: int,
        z_var_dim: int = 8,
        surface_latent_dim: int = 32,
        encoder_hidden_dim: int = 32,
        encoder_dropout: float = 0.05,
        surface_gate_hidden_dim: int = 8,
        surface_gate_mode: GateMode = "baseline_only",
        surface_gate_enabled: bool = True,
        gate_init_open_prob: float = 0.7,
        fusion_hidden_dim: int = 32,
        fusion_dropout: float = 0.05,
        surface_encoder_depth: int = 1,
        fusion_depth: int = 1,
        surface_activation: ActivationName = "relu",
        surface_norm_type: NormType = "layernorm",
        surface_variable_weighting: SurfaceVariableWeighting = "gate",
        surface_ablate_vars: Sequence[str] = (),
    ):
        super().__init__()

        self.surface_feature_dims = {
            name: int(surface_feature_dims[name]) for name in _SURFACE_SLOT_ORDER
        }
        self.surface_native_feature_dims = {
            name: int(surface_native_feature_dims[name]) for name in _SURFACE_SLOT_ORDER
        }
        self.surface_months_per_sample = int(surface_months_per_sample)
        self.n_season_features = int(n_season_features)
        self.z_var_dim = int(z_var_dim)
        self.surface_latent_dim = int(surface_latent_dim)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.gate_hidden_dim = int(surface_gate_hidden_dim)
        self.gate_mode: GateMode = str(surface_gate_mode)
        self.surface_gate_enabled = bool(surface_gate_enabled)
        self.gate_init_open_prob = float(gate_init_open_prob)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.surface_encoder_depth = int(surface_encoder_depth)
        self.fusion_depth = int(fusion_depth)
        self.surface_activation: ActivationName = str(surface_activation)
        self.surface_norm_type: NormType = str(surface_norm_type)
        self.surface_variable_weighting: SurfaceVariableWeighting = str(surface_variable_weighting)

        # Final-method configuration: one physical surface row and (sin, cos) season features.
        if self.surface_months_per_sample != 1 or self.n_season_features != 2:
            raise ValueError("Final M/Q method requires one surface row and two season features")
        if self.gate_mode not in {"baseline_only", "seasonal"}:
            raise ValueError(
                "gate_mode must be 'baseline_only' or 'seasonal', "
                f"got {self.gate_mode!r}"
            )
        if self.surface_variable_weighting not in {"gate", "none", "learned_scalar"}:
            raise ValueError(
                "surface_variable_weighting must be 'gate', 'none', or 'learned_scalar', "
                f"got {self.surface_variable_weighting!r}"
            )
        if self.surface_activation not in {"relu", "gelu", "silu"}:
            raise ValueError(
                "surface_activation must be 'relu', 'gelu', or 'silu', "
                f"got {self.surface_activation!r}"
            )
        if self.surface_norm_type not in {"layernorm", "none"}:
            raise ValueError(
                "surface_norm_type must be 'layernorm' or 'none', "
                f"got {self.surface_norm_type!r}"
            )
        for name in (
            "z_var_dim",
            "surface_latent_dim",
            "encoder_hidden_dim",
            "gate_hidden_dim",
            "fusion_hidden_dim",
            "surface_encoder_depth",
            "fusion_depth",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

        self.encoder_dropout = _validate_dropout("encoder_dropout", encoder_dropout)
        self.fusion_dropout = _validate_dropout("fusion_dropout", fusion_dropout)

        # Feature width per slot must equal native width times months_per_sample.
        for slot_name in _SURFACE_SLOT_ORDER:
            native_width = self.surface_native_feature_dims[slot_name]
            feature_width = self.surface_feature_dims[slot_name]
            expected_width = native_width * self.surface_months_per_sample
            if native_width <= 0:
                raise ValueError(f"{slot_name}: native width must be > 0, got {native_width}")
            if feature_width != expected_width:
                raise ValueError(
                    f"{slot_name}: feature width {feature_width} != native width "
                    f"{native_width} * months {self.surface_months_per_sample}"
                )

        ablate_vars = tuple(str(name) for name in surface_ablate_vars)
        unknown = sorted(set(ablate_vars) - set(_SURFACE_SLOT_ORDER))
        if unknown:
            raise ValueError(f"surface_ablate_vars contains unknown variables: {unknown}")
        if len(set(ablate_vars)) != len(ablate_vars):
            raise ValueError(f"surface_ablate_vars contains duplicates: {ablate_vars}")
        self.surface_ablate_vars = ablate_vars

        # Per-slot encoders map (B, native_width) -> (B, z_var_dim).
        self.encoders = nn.ModuleDict(
            {
                slot_name: _make_latent_mlp(
                    self.surface_native_feature_dims[slot_name],
                    self.encoder_hidden_dim,
                    self.z_var_dim,
                    self.encoder_dropout,
                    depth=self.surface_encoder_depth,
                    activation=self.surface_activation,
                    norm_type=self.surface_norm_type,
                )
                for slot_name in _SURFACE_SLOT_ORDER
            }
        )
        # Base per-slot gate logits (one scalar per slot) initialised near the chosen open probability.
        gate_logits_trainable = not (
            self.surface_variable_weighting == "none"
            or (self.surface_variable_weighting == "gate" and not self.surface_gate_enabled)
        )
        self.gate_base_logits = nn.Parameter(
            torch.full(
                (len(_SURFACE_SLOT_ORDER),),
                _logit_probability(self.gate_init_open_prob),
                dtype=torch.float32,
            ),
            requires_grad=gate_logits_trainable,
        )
        # Optional seasonal gate: a small MLP of season -> per-slot logit offset.
        self.seasonal_gate_mlp = None
        if (
            self.surface_variable_weighting == "gate"
            and self.surface_gate_enabled
            and self.gate_mode == "seasonal"
        ):
            self.seasonal_gate_mlp = nn.Sequential(
                nn.Linear(2, self.gate_hidden_dim),
                _make_activation(self.surface_activation),
                nn.Linear(self.gate_hidden_dim, len(_SURFACE_SLOT_ORDER)),
            )
            # Zero-init the final linear so the seasonal gate starts as the baseline gate.
            final_linear = self.seasonal_gate_mlp[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Fusion MLP maps the concatenated per-slot latents to the shared surface latent.
        self.fusion_mlp = _make_latent_mlp(
            len(_SURFACE_SLOT_ORDER) * self.z_var_dim,
            self.fusion_hidden_dim,
            self.surface_latent_dim,
            self.fusion_dropout,
            depth=self.fusion_depth,
            activation=self.surface_activation,
            norm_type=self.surface_norm_type,
        )

        # Ablation mask (S, 1): zero for every ablated slot, one elsewhere.
        ablate_mask = torch.ones((len(_SURFACE_SLOT_ORDER), 1), dtype=torch.float32)
        for idx, slot_name in enumerate(_SURFACE_SLOT_ORDER):
            if slot_name in self.surface_ablate_vars:
                ablate_mask[idx, 0] = 0.0
        self.register_buffer("_surface_ablate_mask", ablate_mask)

    @staticmethod
    def _validate_2d(
        tensor: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        feature_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Check that tensor is 2-D (batch_size, feature_dim) and move it to dtype/device."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be 2D (B, {feature_dim}), got {tuple(tensor.shape)}")
        if tensor.shape != (batch_size, feature_dim):
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} != expected {(batch_size, feature_dim)}"
            )
        return tensor.to(dtype=dtype, device=device)

    def gate_values_from_monthly_season(self, monthly_season: torch.Tensor) -> torch.Tensor:
        """Return per-slot gate values in [0, 1] from the season (sin, cos) tensor."""
        if monthly_season.ndim != 2 or monthly_season.shape[1] != 2:
            raise ValueError(
                "monthly_season must be shaped (N, 2), got "
                f"{tuple(monthly_season.shape)}"
            )
        monthly_season = monthly_season.to(
            dtype=self.gate_base_logits.dtype,
            device=self.gate_base_logits.device,
        )
        n_rows = int(monthly_season.shape[0])
        if self.surface_variable_weighting == "none":
            return torch.ones(
                (n_rows, len(_SURFACE_SLOT_ORDER)),
                dtype=self.gate_base_logits.dtype,
                device=self.gate_base_logits.device,
            )

        # Base logits per slot; the seasonal MLP adds an offset when enabled.
        logits = self.gate_base_logits.unsqueeze(0).expand(n_rows, len(_SURFACE_SLOT_ORDER))
        if (
            self.surface_variable_weighting == "gate"
            and self.surface_gate_enabled
            and self.gate_mode == "seasonal"
        ):
            if self.seasonal_gate_mlp is None:
                raise RuntimeError("seasonal gate requested but seasonal_gate_mlp is missing")
            logits = logits + self.seasonal_gate_mlp(monthly_season)
        elif self.surface_variable_weighting == "gate" and not self.surface_gate_enabled:
            return torch.ones_like(logits)

        return torch.sigmoid(logits)

    def forward(
        self,
        *,
        ws: torch.Tensor,
        sst: torch.Tensor,
        sss: torch.Tensor,
        ssh: torch.Tensor,
        obp: torch.Tensor,
        season: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Encode each slot, gate it, ablate zeroed slots, fuse into surface_latent (B, surface_latent_dim)."""
        model_device = self.gate_base_logits.device
        model_dtype = self.gate_base_logits.dtype

        if not isinstance(season, torch.Tensor):
            raise TypeError(f"season must be torch.Tensor, got {type(season)}")
        if season.ndim != 2:
            raise ValueError(
                f"season must be 2D (B, {self.n_season_features}), got {tuple(season.shape)}"
            )
        B = int(season.shape[0])
        season = self._validate_2d(
            season,
            name="season",
            batch_size=B,
            feature_dim=self.n_season_features,
            dtype=model_dtype,
            device=model_device,
        )
        surface_inputs = {
            name: self._validate_2d(
                tensor, name=name, batch_size=B,
                feature_dim=self.surface_feature_dims[name],
                dtype=model_dtype, device=model_device,
            )
            for name, tensor in (("ws", ws), ("sst", sst), ("sss", sss),
                                 ("ssh", ssh), ("obp", obp))
        }

        # Per-sample, per-slot gate values in [0, 1], shaped (B, 1, S) for broadcasting.
        gates = self.gate_values_from_monthly_season(season).reshape(
            B, 1, len(_SURFACE_SLOT_ORDER)
        )

        # Encode each slot and apply its gate; keep both the raw and gated latent for diagnostics.
        monthly_latents = []
        gated_monthly_latents = []
        final_latents_before_ablation = []

        for slot_idx, slot_name in enumerate(_SURFACE_SLOT_ORDER):
            x_var = surface_inputs[slot_name]
            z_month = self.encoders[slot_name](x_var).reshape(B, 1, self.z_var_dim)   # (B, 1, z_var_dim)

            gate = gates[:, :, slot_idx].unsqueeze(-1)       # (B, 1, 1)
            z_gated = z_month * gate
            z_final = z_gated[:, 0, :]                       # (B, z_var_dim)

            monthly_latents.append(z_month)
            gated_monthly_latents.append(z_gated)
            final_latents_before_ablation.append(z_final)

        # Apply the ablation mask that zeroes slots listed in surface_ablate_vars (ws in the final method).
        z_final_stack = torch.stack(final_latents_before_ablation, dim=1)              # (B, S, z_var_dim)
        ablate_mask = self._surface_ablate_mask.to(
            dtype=z_final_stack.dtype,
            device=z_final_stack.device,
        )
        z_final_masked = z_final_stack * ablate_mask.unsqueeze(0)                      # (B, S, z_var_dim)

        # Flatten the per-slot latents and fuse into the shared surface latent.
        fusion_input = z_final_masked.reshape(B, len(_SURFACE_SLOT_ORDER) * self.z_var_dim)
        surface_latent = self.fusion_mlp(fusion_input)                                 # (B, surface_latent_dim)
        if torch.isnan(surface_latent).any():
            raise RuntimeError("NaN in surface_latent")

        if not return_diagnostics:
            return surface_latent

        # Diagnostics are detached clones of intermediate tensors; cheap to collect, useful for figures.
        monthly_latent_stack = torch.stack(monthly_latents, dim=2)
        gated_monthly_latent_stack = torch.stack(gated_monthly_latents, dim=2)
        gate_values = gates[:, 0, :]
        diagnostics = {
            "gate_values": gate_values.detach().clone(),
            "monthly_gate_values": gates.detach().clone(),
            "gate_base_logits": self.gate_base_logits.detach().clone(),
            "gate_base_values": torch.sigmoid(self.gate_base_logits.detach()).clone(),
            "monthly_latent_norms_pre_gate": monthly_latent_stack.norm(dim=-1).detach().clone(),
            "monthly_latent_norms_post_gate": gated_monthly_latent_stack.norm(dim=-1).detach().clone(),
            "variable_latent_norms_before_ablation": z_final_stack.norm(dim=-1).detach().clone(),
            "variable_latent_norms_after_ablation": z_final_masked.norm(dim=-1).detach().clone(),
            "surface_ablate_mask": ablate_mask.squeeze(-1).detach().clone(),
            "fusion_input_norm": fusion_input.norm(dim=-1).detach().clone(),
            "surface_latent_norm": surface_latent.norm(dim=-1).detach().clone(),
            "surface_latent": surface_latent.detach().clone(),
            "surface_months_per_sample": self.surface_months_per_sample,
        }
        return surface_latent, diagnostics


class PredictionHead(nn.Module):
    """Configurable final MLP that maps the fused embedding to a scalar prediction.

    head_depth selects 1, 2 or 3 hidden blocks; each block is Linear + activation
    + optional LayerNorm + dropout. Optional residual connections skip around
    blocks when input and output dimensions match.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        head_hidden_dim_1: int = 64,
        head_hidden_dim_2: int = 32,
        head_depth: int = 2,
        head_dropout: float = 0.0,
        head_activation: ActivationName = "relu",
        head_norm_type: NormType = "none",
        head_input_dropout: float = 0.0,
        head_use_residual_mlp: bool = False,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.head_hidden_dim_1 = int(head_hidden_dim_1)
        self.head_hidden_dim_2 = int(head_hidden_dim_2)
        self.head_depth = int(head_depth)
        self.head_dropout = _validate_dropout("head_dropout", head_dropout)
        self.head_input_dropout = _validate_dropout("head_input_dropout", head_input_dropout)
        self.head_activation: ActivationName = str(head_activation)
        self.head_norm_type: NormType = str(head_norm_type)
        self.head_use_residual_mlp = bool(head_use_residual_mlp)

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.head_hidden_dim_1 <= 0:
            raise ValueError(
                f"head_hidden_dim_1 must be > 0, got {self.head_hidden_dim_1}"
            )
        if self.head_hidden_dim_2 <= 0:
            raise ValueError(
                f"head_hidden_dim_2 must be > 0, got {self.head_hidden_dim_2}"
            )
        if self.head_depth not in {1, 2, 3}:
            raise ValueError(f"head_depth must be 1, 2, or 3, got {self.head_depth}")
        if self.head_activation not in {"relu", "gelu", "silu"}:
            raise ValueError(
                "head_activation must be 'relu', 'gelu', or 'silu', "
                f"got {self.head_activation!r}"
            )
        if self.head_norm_type not in {"layernorm", "none"}:
            raise ValueError(
                "head_norm_type must be 'layernorm' or 'none', "
                f"got {self.head_norm_type!r}"
            )

        # Hidden widths: first block uses head_hidden_dim_1, later blocks use head_hidden_dim_2.
        hidden_dims = [self.head_hidden_dim_1]
        if self.head_depth >= 2:
            hidden_dims.append(self.head_hidden_dim_2)
        if self.head_depth >= 3:
            hidden_dims.append(self.head_hidden_dim_2)

        self.input_dropout = nn.Dropout(p=self.head_input_dropout)
        self.hidden_layers = nn.ModuleList()
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    _make_activation(self.head_activation),
                    _make_norm(self.head_norm_type, hidden_dim),
                    nn.Dropout(p=self.head_dropout),
                )
            )
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP and return a (B,) scalar prediction."""
        out = self.input_dropout(x)
        for layer in self.hidden_layers:
            residual = out
            out = layer(out)
            # Add the residual only when shapes match (first block typically changes the width).
            if self.head_use_residual_mlp and out.shape == residual.shape:
                out = out + residual
        return self.output_layer(out).squeeze(dim=-1)


class ProfileModelBlockedKFold2Latent(nn.Module):
    """Latent profile model combining a GNN over Argo compartments with a surface latent.

    Pipeline per batch:
      1. Argo profiles are soft-assigned to C compartments based on their
         (lon, lat, days) coordinates (DistanceAssignerModule).
      2. Each compartment is summarised by an assignment-weighted mean of
         the profile features, then processed by a per-compartment MLP
         (optionally gated), yielding node embeddings on the compartment graph.
      3. A graph adjacency is built as a convex mix of a geometric prior
         (compartment-centre distances) and a learned low-rank prior, then
         thresholded and row-normalised.
      4. A ChebConv GNN propagates information between compartments.
      5. A SurfaceLatentModule turns SST, SSS, SSH and OBP (wind is zeroed)
         into a shared surface latent via per-variable encoders and gating.
      6. The prediction head consumes the flattened GNN output, the surface
         latent, and optionally the Argo-batch mean and the season angle.
    """

    def __init__(
        self,
        n_features: int,
        n_compartments: int = 16,
        n_embedding: int = 32,
        *,
        node_assigner: DistanceAssignerModule | None = None,
        argo_mean_in_embedding: bool = False,
        argo_mean_in_gnn_input: bool = True,
        assignment_threshold: float = 0.05,
        embedding_dropout: float = 0.22,
        adj_threshold: float = 0.1,
        num_gcn_layers: int = 3,
        adjacency_embedding_dim: int = 8,
        cheb_k: int = 3,
        gcn_normalization: GCNNormalization = "rw",
        compartment_hidden_mult: float = 2.0,
        compartment_gate_hidden_mult: float = 1.0,
        compartment_gate_enabled: bool = True,
        n_ws_features: int = 10,
        n_sst_features: int = 9,
        n_sss_features: int = 10,
        n_ssh_features: int = 9,
        n_obp_features: int = 10,
        n_season_features: int = 2,
        surface_native_feature_dims: Dict[str, int] | None = None,
        surface_months_per_sample: int = 1,
        z_var_dim: int = 8,
        surface_latent_dim: int = 32,
        encoder_hidden_dim: int = 32,
        encoder_dropout: float = 0.05,
        surface_gate_hidden_dim: int = 8,
        surface_gate_mode: GateMode = "baseline_only",
        surface_gate_enabled: bool = True,
        gate_init_open_prob: float = 0.7,
        fusion_hidden_dim: int = 32,
        fusion_dropout: float = 0.05,
        surface_encoder_depth: int = 1,
        fusion_depth: int = 1,
        surface_activation: ActivationName = "relu",
        surface_norm_type: NormType = "layernorm",
        surface_variable_weighting: SurfaceVariableWeighting = "gate",
        surface_ablate_vars: Sequence[str] = (),
        use_season_in_head: bool = False,
        return_surface_diagnostics: bool = False,
        sigmas_learnable: bool = True,
        sigma_policy: SigmaPolicy = "all_learnable",
        centers_learnable: bool = True,
        alpha_init: float = 0.5,
        alpha_learnable: bool = False,
        head_hidden_dim_1: int = 64,
        head_hidden_dim_2: int = 32,
        head_depth: int = 2,
        head_dropout: float = 0.0,
        head_activation: ActivationName = "relu",
        head_norm_type: NormType = "none",
        head_input_dropout: float = 0.0,
        head_use_residual_mlp: bool = False,
    ):
        super().__init__()

        # --- Store configuration ---
        self.n_compartments = int(n_compartments)
        self.n_features = int(n_features)
        self.n_embedding = int(n_embedding)
        self.argo_mean_in_embedding = bool(argo_mean_in_embedding)
        self.argo_mean_in_gnn_input = bool(argo_mean_in_gnn_input)
        self.assignment_threshold = float(assignment_threshold)
        self.adj_threshold = float(adj_threshold)
        self.adjacency_embedding_dim = int(adjacency_embedding_dim)
        self.cheb_k = int(cheb_k)
        self.gcn_normalization: GCNNormalization = str(gcn_normalization)
        self.compartment_hidden_mult = float(compartment_hidden_mult)
        self.compartment_gate_hidden_mult = float(compartment_gate_hidden_mult)
        self.compartment_gate_enabled = bool(compartment_gate_enabled)
        self.centers_learnable = bool(centers_learnable)

        self.n_ws_features = int(n_ws_features)
        self.n_sst_features = int(n_sst_features)
        self.n_sss_features = int(n_sss_features)
        self.n_ssh_features = int(n_ssh_features)
        self.n_obp_features = int(n_obp_features)
        self.n_season_features = int(n_season_features)
        if self.n_season_features <= 0 or self.n_season_features % 2 != 0:
            raise ValueError(
                "n_season_features must be a positive even number of "
                f"monthly sin/cos values, got {self.n_season_features}"
            )

        self.use_season_in_head = bool(use_season_in_head)
        self.return_surface_diagnostics = bool(return_surface_diagnostics)
        self._last_surface_diagnostics = None

        self.surface_feature_dims = {
            "ws": self.n_ws_features,
            "sst": self.n_sst_features,
            "sss": self.n_sss_features,
            "ssh": self.n_ssh_features,
            "obp": self.n_obp_features,
        }
        self.surface_months_per_sample = int(surface_months_per_sample)
        if self.surface_months_per_sample != 1:
            raise ValueError("Final M/Q method requires one physical surface row")
        if surface_native_feature_dims is None:
            surface_native_feature_dims = self.surface_feature_dims
        self.surface_native_feature_dims = {
            name: int(surface_native_feature_dims[name]) for name in _SURFACE_SLOT_ORDER
        }

        self.alpha_learnable = bool(alpha_learnable)
        self.graph_backend = "pyg"
        self.sigma_policy: SigmaPolicy = str(sigma_policy)

        # --- Validate configuration ---
        if self.n_compartments <= 0:
            raise ValueError(f"n_compartments must be > 0, got {self.n_compartments}")
        if self.n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {self.n_features}")
        if self.n_embedding <= 0:
            raise ValueError(f"n_embedding must be > 0, got {self.n_embedding}")
        if self.adjacency_embedding_dim <= 0:
            raise ValueError(
                f"adjacency_embedding_dim must be > 0, got {self.adjacency_embedding_dim}"
            )
        if self.cheb_k <= 0:
            raise ValueError(f"cheb_k must be > 0, got {self.cheb_k}")
        if self.gcn_normalization not in {"rw", "sym", "none"}:
            raise ValueError(
                "gcn_normalization must be 'rw', 'sym', or 'none', "
                f"got {self.gcn_normalization!r}"
            )
        if self.compartment_hidden_mult <= 0.0:
            raise ValueError(
                f"compartment_hidden_mult must be > 0, got {self.compartment_hidden_mult}"
            )
        if self.compartment_gate_hidden_mult <= 0.0:
            raise ValueError(
                "compartment_gate_hidden_mult must be > 0, "
                f"got {self.compartment_gate_hidden_mult}"
            )
        if not (0.0 <= self.assignment_threshold <= 1.0):
            raise ValueError(
                f"assignment_threshold must lie in [0, 1], got {self.assignment_threshold}"
            )
        if not (0.0 <= self.adj_threshold <= 1.0):
            raise ValueError(f"adj_threshold must lie in [0, 1], got {self.adj_threshold}")

        # --- Submodules: compartment MLPs and gates ---
        compartment_hidden_dim = _scaled_hidden_dim(
            self.n_embedding,
            self.compartment_hidden_mult,
            name="compartment_hidden_mult",
        )
        compartment_gate_hidden_dim = _scaled_hidden_dim(
            self.n_embedding,
            self.compartment_gate_hidden_mult,
            name="compartment_gate_hidden_mult",
        )
        self.compartment_hidden_dim = int(compartment_hidden_dim)
        self.compartment_gate_hidden_dim = int(compartment_gate_hidden_dim)

        # One tanh-bounded MLP per compartment: feature mean + (lon, lat, days) -> embedding.
        self.compartment_models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.n_features + 3, self.compartment_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.compartment_hidden_dim, self.n_embedding),
                    nn.Tanh(),
                )
                for _ in range(self.n_compartments)
            ]
        )

        # Optional per-compartment sigmoid gate that scales the embedding element-wise.
        self.compartment_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.n_features + 3, self.compartment_gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.compartment_gate_hidden_dim, self.n_embedding),
                    nn.Sigmoid(),
                )
                for _ in range(self.n_compartments)
            ]
            if self.compartment_gate_enabled
            else []
        )

        # Learned low-rank factors used to build a data-driven adjacency prior.
        self.E1 = nn.Parameter(
            torch.randn(self.n_compartments, self.adjacency_embedding_dim),
            requires_grad=True,
        )
        self.E2 = nn.Parameter(
            torch.randn(self.n_compartments, self.adjacency_embedding_dim),
            requires_grad=True,
        )

        # Graph convolution stack over the compartment graph.
        self.gcn = BysGNN_GNN_GCN(
            self.n_embedding,
            conv_hidden_dim=self.n_embedding,
            conv_num_layers=int(num_gcn_layers),
            cheb_k=self.cheb_k,
            gcn_normalization=self.gcn_normalization,
        )

        # Node assigner: owns compartment centres and Gaussian widths.
        if node_assigner is not None:
            self.node_assigner = node_assigner
        else:
            self.node_assigner = DistanceAssignerModule(
                self.n_compartments,
                centers_learnable=self.centers_learnable,
                sigmas_learnable=bool(sigmas_learnable),
                sigma_policy=self.sigma_policy,
                dtype=torch.float32,
            )
        self.sigma_policy = self.node_assigner.sigma_policy
        self.centers_learnable = bool(self.node_assigner.centers.requires_grad)

        # alpha mixes the geometric and the learned adjacency: adj = alpha * geom + (1 - alpha) * learned.
        if self.alpha_learnable:
            if not (0.0 < float(alpha_init) < 1.0):
                raise ValueError(
                    f"alpha_init must lie strictly inside (0, 1) when alpha is learnable; "
                    f"got {alpha_init}"
                )
            self._alpha_logit = nn.Parameter(
                torch.tensor(log(float(alpha_init) / (1.0 - float(alpha_init))), dtype=torch.float32),
                requires_grad=True,
            )
            self.register_buffer("_alpha_fixed", torch.tensor(0.0, dtype=torch.float32))
        else:
            if not (0.0 <= float(alpha_init) <= 1.0):
                raise ValueError(
                    f"alpha_init must lie in [0, 1] when alpha is fixed; got {alpha_init}"
                )
            self.register_buffer("_alpha_fixed", torch.tensor(float(alpha_init), dtype=torch.float32))
            self._alpha_logit = None

        self.embedding_dropout = nn.Dropout(p=float(embedding_dropout))
        # Surface branch: SST/SSS/SSH/OBP (and the inactive WS slot).
        self.surface_module = SurfaceLatentModule(
            surface_feature_dims=self.surface_feature_dims,
            surface_native_feature_dims=self.surface_native_feature_dims,
            surface_months_per_sample=self.surface_months_per_sample,
            n_season_features=self.n_season_features,
            z_var_dim=int(z_var_dim),
            surface_latent_dim=int(surface_latent_dim),
            encoder_hidden_dim=int(encoder_hidden_dim),
            encoder_dropout=float(encoder_dropout),
            surface_gate_hidden_dim=int(surface_gate_hidden_dim),
            surface_gate_mode=surface_gate_mode,
            surface_gate_enabled=bool(surface_gate_enabled),
            gate_init_open_prob=float(gate_init_open_prob),
            fusion_hidden_dim=int(fusion_hidden_dim),
            fusion_dropout=float(fusion_dropout),
            surface_encoder_depth=int(surface_encoder_depth),
            fusion_depth=int(fusion_depth),
            surface_activation=surface_activation,
            surface_norm_type=surface_norm_type,
            surface_variable_weighting=surface_variable_weighting,
            surface_ablate_vars=surface_ablate_vars,
        )
        self.surface_latent_dim = int(surface_latent_dim)
        self.surface_ablate_vars = tuple(self.surface_module.surface_ablate_vars)

        # Head input: optional argo_mean + flattened GNN + surface_latent + optional season.
        self.head_input_dim = (
            (self.n_features if self.argo_mean_in_embedding else 0)
            + self.n_embedding * self.n_compartments
            + self.surface_latent_dim
            + (self.n_season_features if self.use_season_in_head else 0)
        )

        self.prediction_head = PredictionHead(
            self.head_input_dim,
            head_hidden_dim_1=int(head_hidden_dim_1),
            head_hidden_dim_2=int(head_hidden_dim_2),
            head_depth=int(head_depth),
            head_dropout=float(head_dropout),
            head_activation=head_activation,
            head_norm_type=head_norm_type,
            head_input_dropout=float(head_input_dropout),
            head_use_residual_mlp=bool(head_use_residual_mlp),
        )
        self.head_hidden_dim_1 = int(head_hidden_dim_1)
        self.head_hidden_dim_2 = int(head_hidden_dim_2)
        self.head_depth = int(head_depth)
        self.head_dropout = float(head_dropout)
        self.head_activation: ActivationName = str(head_activation)
        self.head_norm_type: NormType = str(head_norm_type)
        self.head_input_dropout = float(head_input_dropout)
        self.head_use_residual_mlp = bool(head_use_residual_mlp)

    @property
    def uses_torch_geometric(self) -> bool:
        return self.gcn.uses_torch_geometric

    @staticmethod
    def _assert_tensor(name: str, value) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(value)}")

    @staticmethod
    def _validate_and_move_2d_feature(
        tensor: torch.Tensor,
        name: str,
        batch_size: int,
        feat_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Check that tensor is 2-D (batch_size, feat_dim) and move it to dtype/device."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be 2D (B, {feat_dim}), got {tuple(tensor.shape)}")
        if tensor.shape != (batch_size, feat_dim):
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} != expected {(batch_size, feat_dim)}"
            )
        return tensor.to(dtype=dtype, device=device)

    def alpha_value(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return the current alpha mixing coefficient in [0, 1]."""
        if self.alpha_learnable:
            return torch.sigmoid(self._alpha_logit).to(dtype=dtype, device=device)
        return self._alpha_fixed.to(dtype=dtype, device=device)

    def _build_adjacency(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Build the compartment adjacency as alpha * geometric + (1 - alpha) * learned.

        Weak edges below adj_threshold are dropped; the resulting matrix is
        row-normalised. Isolated rows (no edges) fall back to a self-loop.
        """
        # Learned prior: low-rank similarity E1 E2^T, ReLU-rectified, then softmax per row.
        similarity = torch.mm(self.E1, self.E2.permute(1, 0))
        similarity = torch.relu(similarity)
        learned_matrix = torch.softmax(similarity, dim=-1)

        # Geometric prior: compartment-centre pairwise distances, softmax per row.
        geometric_matrix = self.node_assigner.get_adjacent_matrix().detach()
        geometric_matrix = geometric_matrix.to(
            dtype=learned_matrix.dtype,
            device=learned_matrix.device,
        )

        alpha = self.alpha_value(dtype=learned_matrix.dtype, device=learned_matrix.device)
        adjacency_matrix = alpha * geometric_matrix + (1.0 - alpha) * learned_matrix

        # Threshold weak edges to encourage a sparser graph.
        adjacency_matrix = torch.where(
            adjacency_matrix < self.adj_threshold,
            torch.zeros_like(adjacency_matrix),
            adjacency_matrix,
        )
        # Row-normalise; rows that lost every edge fall back to a self-loop.
        row_sum = adjacency_matrix.sum(dim=-1, keepdim=True)
        identity = torch.eye(
            self.n_compartments,
            dtype=adjacency_matrix.dtype,
            device=adjacency_matrix.device,
        )
        adjacency_matrix = torch.where(
            row_sum <= 1e-8,
            identity,
            adjacency_matrix / row_sum.clamp_min(1e-8),
        )

        return adjacency_matrix.to(dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask_padded: torch.Tensor,
        lon: torch.Tensor,
        lat: torch.Tensor,
        days: torch.Tensor,
        ws: torch.Tensor,
        sst: torch.Tensor,
        sss: torch.Tensor,
        ssh: torch.Tensor,
        obp: torch.Tensor,
        season: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return the scalar prediction and (GNN output, flattened head input) for diagnostics."""
        # --- Validate and move the Argo-profile tensors ---
        self._assert_tensor("x", x)
        self._assert_tensor("mask_padded", mask_padded)
        self._assert_tensor("lon", lon)
        self._assert_tensor("lat", lat)
        self._assert_tensor("days", days)

        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        x = x.to(dtype=model_dtype, device=model_device)
        mask_padded = mask_padded.to(dtype=model_dtype, device=model_device)
        lon = lon.to(dtype=model_dtype, device=model_device)
        lat = lat.to(dtype=model_dtype, device=model_device)
        days = days.to(dtype=model_dtype, device=model_device)

        if x.ndim != 3:
            raise ValueError(f"x must be (B, P, F), got {tuple(x.shape)}")
        if mask_padded.ndim != 2:
            raise ValueError(f"mask_padded must be (B, P), got {tuple(mask_padded.shape)}")
        if lon.ndim != 2 or lat.ndim != 2 or days.ndim != 2:
            raise ValueError(
                f"lon/lat/days must be (B, P); got {tuple(lon.shape)}, {tuple(lat.shape)}, {tuple(days.shape)}"
            )

        B, P, F_in = x.shape
        if F_in != self.n_features:
            raise ValueError(f"x feature mismatch: got {F_in}, expected {self.n_features}")
        if mask_padded.shape != (B, P):
            raise ValueError(f"mask_padded mismatch: {tuple(mask_padded.shape)} vs {(B, P)}")
        if lon.shape != (B, P) or lat.shape != (B, P) or days.shape != (B, P):
            raise ValueError("lon/lat/days shapes must match the main profile shape")

        # --- Validate and move the surface and season tensors ---
        ws = self._validate_and_move_2d_feature(
            ws, "ws", B, self.n_ws_features, model_dtype, model_device
        )
        sst = self._validate_and_move_2d_feature(
            sst, "sst", B, self.n_sst_features, model_dtype, model_device
        )
        sss = self._validate_and_move_2d_feature(
            sss, "sss", B, self.n_sss_features, model_dtype, model_device
        )
        ssh = self._validate_and_move_2d_feature(
            ssh, "ssh", B, self.n_ssh_features, model_dtype, model_device
        )
        obp = self._validate_and_move_2d_feature(
            obp, "obp", B, self.n_obp_features, model_dtype, model_device
        )
        season = self._validate_and_move_2d_feature(
            season, "season", B, self.n_season_features, model_dtype, model_device
        )

        # --- Argo pooling: mean over valid padded profiles ---
        profile_mask = mask_padded.unsqueeze(-1)                                        # (B, P, 1)
        x_masked = x * profile_mask
        valid_counts = torch.sum(profile_mask, dim=1)                                   # (B, 1)
        valid_counts_safe = torch.clamp(valid_counts, min=1.0)
        argo_mean = torch.sum(x_masked, dim=1) / valid_counts_safe                      # (B, F)
        argo_mean = torch.where(valid_counts > 0, argo_mean, torch.zeros_like(argo_mean))

        # Optionally subtract the batch mean from each profile before feeding the GNN branch.
        if not self.argo_mean_in_gnn_input:
            x = x - argo_mean.unsqueeze(1)
            x = x * profile_mask

        # --- Soft-assign each profile to a compartment ---
        assignment_input = torch.stack([lon, lat, days], dim=-1)                        # (B, P, 3)
        node_assignments = self.node_assigner(assignment_input)                         # (B, P, C) probabilities
        if torch.isnan(node_assignments).any():
            raise RuntimeError("NaN in fresh node_assignments")

        # Mask padded positions and threshold weak assignments (keep sparser attribution).
        node_assignments = node_assignments * profile_mask
        assign_mask = node_assignments >= self.assignment_threshold
        node_assignments = node_assignments * assign_mask.to(dtype=node_assignments.dtype)
        if torch.isnan(node_assignments).any():
            raise RuntimeError("NaN after assignment thresholding")

        # Augment the profile features with their (lon, lat, days) before the per-compartment pooling.
        input_to_layer = torch.cat(
            (
                x,
                lon.unsqueeze(-1),
                lat.unsqueeze(-1),
                days.unsqueeze(-1),
            ),
            dim=-1,
        )

        # --- Per-compartment assignment-weighted feature mean ---
        assignment_mass = torch.sum(node_assignments, dim=1)                            # (B, C) total mass
        compartment_presence = (assignment_mass > 1e-6).to(dtype=model_dtype).unsqueeze(-1)   # (B, C, 1) active-compartment flag

        weighted_mean = torch.sum(
            input_to_layer.unsqueeze(-2) * node_assignments.unsqueeze(-1),
            dim=1,
        ) / (assignment_mass.unsqueeze(-1) + 1e-6)                                      # (B, C, F+3)
        weighted_mean = weighted_mean * compartment_presence
        if torch.isnan(weighted_mean).any():
            raise RuntimeError("NaN in weighted_mean")
        if self.n_compartments != weighted_mean.shape[1]:
            raise RuntimeError("n_compartments mismatch inside weighted_mean")

        # --- Per-compartment MLP (and optional sigmoid gate) ---
        compartment_embeddings = []
        for i in range(self.n_compartments):
            compartment_embedding = self.compartment_models[i](weighted_mean[:, i])
            if self.compartment_gate_enabled:
                compartment_embedding = compartment_embedding * self.compartment_gate[i](
                    weighted_mean[:, i]
                )
            compartment_embeddings.append(compartment_embedding.unsqueeze(1))

        graphs = torch.cat(compartment_embeddings, dim=1)                               # (B, C, n_embedding)
        graphs = graphs * compartment_presence
        if torch.isnan(graphs).any():
            raise RuntimeError("NaN in graphs")

        # --- GNN over the compartment graph ---
        adjacency_matrix = self._build_adjacency(dtype=graphs.dtype, device=graphs.device)
        X_gnn = self.gcn(graphs, adjacency_matrix, node_mask=compartment_presence)      # (B, C, n_embedding)
        X_hat = self.embedding_dropout(X_gnn) * compartment_presence
        if torch.isnan(X_hat.reshape(X_hat.shape[0], -1)).any():
            raise RuntimeError("NaN in X_hat")

        # --- Surface branch: SST, SSS, SSH, OBP (and the inactive WS slot) ---
        surface_result = self.surface_module(
            ws=ws,
            sst=sst,
            sss=sss,
            ssh=ssh,
            obp=obp,
            season=season,
            return_diagnostics=self.return_surface_diagnostics,
        )
        if self.return_surface_diagnostics:
            surface_latent, self._last_surface_diagnostics = surface_result
        else:
            surface_latent = surface_result
            self._last_surface_diagnostics = None
        if surface_latent.shape != (B, self.surface_latent_dim):
            raise RuntimeError(
                f"surface_latent shape {tuple(surface_latent.shape)} != "
                f"{(B, self.surface_latent_dim)}"
            )

        # --- Head input: concatenate the enabled branches ---
        head_blocks = []
        if self.argo_mean_in_embedding:
            head_blocks.append(argo_mean)
        head_blocks.extend(
            [
                X_hat.reshape(X_hat.shape[0], -1),      # flattened GNN output
                surface_latent,                         # (B, surface_latent_dim)
            ]
        )
        if self.use_season_in_head:
            head_blocks.append(season)

        embedding_all_profiles = torch.cat(head_blocks, dim=-1)
        if embedding_all_profiles.shape[-1] != self.head_input_dim:
            raise RuntimeError(
                f"Head input mismatch: got {embedding_all_profiles.shape[-1]}, "
                f"expected {self.head_input_dim}"
            )
        if torch.isnan(embedding_all_profiles).any():
            raise RuntimeError("NaN in embedding_all_profiles")

        # --- Final MLP produces the scalar prediction ---
        out = self.prediction_head(embedding_all_profiles)

        return out, (X_hat, embedding_all_profiles)

    def surface_forward_with_diagnostics(
        self,
        ws: torch.Tensor,
        sst: torch.Tensor,
        sss: torch.Tensor,
        ssh: torch.Tensor,
        obp: torch.Tensor,
        season: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run only the surface module and return detailed diagnostics."""
        surface_latent, diagnostics = self.surface_module(
            ws=ws,
            sst=sst,
            sss=sss,
            ssh=ssh,
            obp=obp,
            season=season,
            return_diagnostics=True,
        )
        return surface_latent, diagnostics

    def last_surface_diagnostics(self):
        """Return diagnostics captured by the last full forward, if enabled."""
        return self._last_surface_diagnostics


# Final HPO-selected configuration and four-surface input contract.
MODEL_KWARGS = {
    "n_compartments": 26, "n_embedding": 40,
    "argo_mean_in_embedding": True, "argo_mean_in_gnn_input": True,
    "assignment_threshold": 0.08, "embedding_dropout": 0.22,
    "adj_threshold": 0.15, "num_gcn_layers": 1,
    "adjacency_embedding_dim": 8, "cheb_k": 4,
    "gcn_normalization": "sym", "compartment_hidden_mult": 2.0,
    "compartment_gate_hidden_mult": 1.0, "compartment_gate_enabled": True,
    "z_var_dim": 4, "surface_latent_dim": 24,
    "encoder_hidden_dim": 32, "encoder_dropout": 0.19502225407155796,
    "surface_gate_hidden_dim": 16, "surface_gate_mode": "seasonal",
    "surface_gate_enabled": True, "gate_init_open_prob": 0.6,
    "fusion_hidden_dim": 64, "fusion_dropout": 0.12331468451116898,
    "surface_encoder_depth": 2, "fusion_depth": 2,
    "surface_activation": "silu",
    "surface_norm_type": "layernorm", "surface_variable_weighting": "gate",
    "use_season_in_head": True, "return_surface_diagnostics": False,
    "sigmas_learnable": True, "sigma_policy": "all_learnable",
    "centers_learnable": True, "alpha_init": 0.5,
    "alpha_learnable": True, "head_hidden_dim_1": 32,
    "head_hidden_dim_2": 32, "head_depth": 2,
    "head_dropout": 0.18621593640966735,
    "head_activation": "relu", "head_norm_type": "none",
    "head_input_dropout": 0.25593472918369947,
    "head_use_residual_mlp": False,
}
SIGMA_INIT = 0.03915092933365262       # HPO-selected initial Gaussian width
SURFACES = ("sst", "sss", "ssh", "obp")   # four public surface predictors; wind is zeroed internally


def compute_initial_centers(train_dataset) -> np.ndarray:
    """Fit 26 KMeans centres on the (lon, lat, days) coordinates of valid training profiles."""
    coordinates = np.stack(
        [np.asarray(getattr(train_dataset, name), dtype=np.float32).ravel()
         for name in ("lon_scaled", "lat_scaled", "days_scaled")],
        axis=-1,
    )
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if len(coordinates) < 26:
        raise ValueError("Too few valid training profiles for 26 KMeans centers")
    return KMeans(n_clusters=26, n_init="auto", random_state=0).fit(
        coordinates
    ).cluster_centers_.astype(np.float32)


class PaperModel(ProfileModelBlockedKFold2Latent):
    """Final-method wrapper: callers pass a batch dict with the four public surface predictors.

    The checkpoint retains an inactive wind slot: this subclass supplies a zero
    wind tensor internally so saved weights still load.
    """

    def forward(self, batch: Mapping[str, torch.Tensor]):
        sst = batch["sst"]
        inactive_wind = sst.new_zeros((sst.shape[0], self.n_ws_features))   # zero WS slot preserved by the checkpoint
        return super().forward(
            batch["x"], batch["mask_padded"], batch["lon"], batch["lat"],
            batch["days"], inactive_wind, sst, batch["sss"], batch["ssh"],
            batch["obp"], batch["season"],
        )


def build_model(train_dataset, *, initialize_assignment: bool = True,
                center_init: str = "kmeans") -> PaperModel:
    """Instantiate the final-method model with the HPO-selected hyperparameters for one fold."""
    stats = train_dataset.norm_stats
    bins = int(stats["surface_n_bins"])
    if stats["surface_feature_dims"] != {name: bins for name in SURFACES}:
        raise ValueError("Expected one normalized bin row for each paper surface")
    if stats["surface_months_per_sample"] != 1 or stats["season_feature_dim"] != 2:
        raise ValueError("Expected monthly M or quarter-mean Q inputs")
    # Compartment centres: either from the frozen reference (.npz) or fitted fresh with KMeans.
    assigner = None
    if initialize_assignment:
        if center_init == "historical":
            with np.load(Path(__file__).with_name("initial_centers.npz")) as reference:
                centers = reference[f"{stats['regime']}_{int(stats['fold_number']):02d}"].copy()
        elif center_init == "kmeans":
            centers = compute_initial_centers(train_dataset)
        else:
            raise ValueError("center_init must be 'historical' or 'kmeans'")
        if centers.shape != (26, 3) or not np.isfinite(centers).all():
            raise ValueError("Invalid paper KMeans initialization centers")
        assigner = DistanceAssignerModule(
            n_compartments=26, centers=centers, centers_learnable=True,
            sigmas_learnable=True, sigma_policy="all_learnable", dtype=torch.float32,
        )
    model = PaperModel(
        n_features=int(train_dataset.F), n_ws_features=bins,
        n_sst_features=bins, n_sss_features=bins,
        n_ssh_features=bins, n_obp_features=bins,
        n_season_features=2,
        surface_native_feature_dims={name: bins for name in ("ws", *SURFACES)},
        surface_months_per_sample=1, node_assigner=assigner,
        surface_ablate_vars=("ws",), **MODEL_KWARGS,
    )
    model.node_assigner.apply_sigma_init(SIGMA_INIT)
    # Attach the data contract so load_checkpoint can refuse mismatched checkpoints.
    model.dataset_contract = {
        "regime": stats["regime"], "fold": int(stats["fold_number"]),
        "surface_n_bins": bins, "surface_lat_rows": int(stats["surface_lat_rows"]),
        "surface_temporal_mode": stats["surface_temporal_mode"],
    }
    model.normalization = stats
    return model


def load_checkpoint(model: PaperModel, path: str | Path) -> dict:
    """Validate a checkpoint against the current data contract and load its state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    # Reject checkpoints that were trained with the WS slot active or with any ablation enabled.
    if config["surface_include_ws"] or config.get("ablate_ws", False) or any(
        config.get("ablate_" + name, False)
        for name in (*SURFACES, "argo_gnn", "argo_mean", "season_all", "season_gate", "season_head")
    ):
        raise ValueError("Checkpoint does not use the final, unablated paper method")
    # Dataset contract: regime, fold, bins/rows and surface temporal mode must match.
    for key, expected in model.dataset_contract.items():
        if key == "surface_lat_rows" and key not in config:
            from .dataset import SCALES

            actual = SCALES.get(config["surface_n_bins"])
        else:
            actual = config[key]
        if actual != expected:
            raise ValueError(f"Checkpoint {key}={actual!r}, expected {expected!r}")
    if checkpoint["norm_stats"]["locked_hpo_trial"] != "HPO1_M/trial_0340":
        raise ValueError("Checkpoint is not from the locked final architecture")
    # Normalisation stats (target, Argo, surfaces) must match within a tight tolerance.
    saved = checkpoint["norm_stats"]
    current = model.normalization
    for key in ("y_mean_train", "y_std_train"):
        if not np.isclose(current[key], saved[key], rtol=0, atol=1e-5):
            raise ValueError(f"Checkpoint {key} differs from current data")
    for group, fields in (
        ("argo_norm_means", ("rho",)), ("argo_norm_stds", ("rho",)),
        ("surface_norm_means", SURFACES), ("surface_norm_stds", SURFACES),
    ):
        for field in fields:
            if not np.allclose(current[group][field], saved[group][field], rtol=0, atol=1e-5):
                raise ValueError(f"Checkpoint {group}.{field} differs from current data")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint
