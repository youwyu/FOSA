"""Language-conditioned history policy used by DAgger."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class HistoryPolicy(nn.Module):
    """Task text + fixed-length actor-observation history -> continuous action.

    Text arrives as frozen pretrained per-token features (see
    ``dagger_lib.text.TaskTextEncoder``) so unseen task descriptions land in
    the same semantic space the policy was trained on.
    """

    def __init__(
        self,
        *,
        observation_size: int,
        action_size: int,
        history_horizon: int,
        text_dim: int = 896,
        max_text_tokens: int = 48,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.history_horizon = int(history_horizon)
        self.max_text_tokens = int(max_text_tokens)
        self.model_config = {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "history_horizon": self.history_horizon,
            "text_dim": int(text_dim),
            "max_text_tokens": self.max_text_tokens,
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "dropout": float(dropout),
        }

        self.state_projection = nn.Linear(self.observation_size, hidden_dim)
        self.text_projection = nn.Linear(int(text_dim), hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1 + self.history_horizon + self.max_text_tokens, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.action_size),
        )
        self.register_buffer("observation_mean", torch.zeros(self.observation_size))
        self.register_buffer("observation_std", torch.ones(self.observation_size))
        self.register_buffer("action_mean", torch.zeros(self.action_size))
        self.register_buffer("action_std", torch.ones(self.action_size))
        self.register_buffer("normalizers_ready", torch.tensor(False))

    def set_normalizers(
        self,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.observation_mean.copy_(observation_mean)
        self.observation_std.copy_(observation_std.clamp_min(1.0e-5))
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std.clamp_min(1.0e-5))
        self.normalizers_ready.fill_(True)

    def forward(
        self,
        histories: torch.Tensor,
        text_features: torch.Tensor,
        text_padding: torch.Tensor,
        text_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = (histories - self.observation_mean) / self.observation_std
        state_tokens = self.state_projection(normalized)
        text_features = text_features[:, : self.max_text_tokens]
        text_padding = text_padding[:, : self.max_text_tokens]
        language = self.text_projection(text_features)
        if text_indices is not None:
            language = language[text_indices]
            text_padding = text_padding[text_indices]
        cls = self.cls_token.expand(histories.shape[0], -1, -1).contiguous()
        tokens = torch.cat((cls, state_tokens, language), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        mask = torch.cat(
            (
                torch.zeros(histories.shape[0], 1 + self.history_horizon, dtype=torch.bool, device=histories.device),
                text_padding,
            ),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)[:, 0]
        return self.action_head(encoded)

    @torch.inference_mode()
    def act_tensor(
        self,
        histories: np.ndarray | torch.Tensor,
        text_features: torch.Tensor,
        text_padding: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        history_tensor = torch.as_tensor(histories, dtype=torch.float32, device=device)
        features = text_features.to(device)
        padding = text_padding.to(device)
        # A single task text is shared by every history in the batch.
        text_indices = torch.zeros(len(history_tensor), dtype=torch.long, device=device) if len(features) == 1 else None
        action = self(history_tensor, features, padding, text_indices) * self.action_std + self.action_mean
        self.train(was_training)
        return action.clamp(-1.0, 1.0)

    def act(
        self,
        histories: np.ndarray | torch.Tensor,
        text_features: torch.Tensor,
        text_padding: torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        return self.act_tensor(histories, text_features, text_padding, device).cpu().numpy()
