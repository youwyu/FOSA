"""Frozen Qwen2.5-0.5B text features for task conditioning (MiniVLA's language backbone, used without the rest of MiniVLA)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple, Sequence

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils.logging import disable_progress_bar, set_verbosity_error

TEXT_BACKBONE = "Qwen/Qwen2.5-0.5B"


class FrozenTextEncoder:
    """Frozen Qwen text encoder shared by the student policy and FOSA's program encoder."""

    def __init__(self, max_length: int = 128, device: str | torch.device = "cpu") -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        disable_progress_bar()
        set_verbosity_error()
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE)
        self.model = AutoModel.from_pretrained(TEXT_BACKBONE).eval()
        self.device = torch.device(device)
        self.model.to(self.device).requires_grad_(False)
        self.max_length = int(max_length)
        self.text_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode_tokens(self, texts: Sequence[str], *, fixed_length: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            list(texts),
            padding="max_length" if fixed_length else True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        attention = tokens["attention_mask"].bool()
        hidden = self.model(**tokens).last_hidden_state.float()
        return hidden * attention.unsqueeze(-1), attention


class TaskText(NamedTuple):
    """Per-token text features and their padding mask (True marks padding)."""

    features: torch.Tensor
    padding: torch.Tensor


def load_task_text(path: Path) -> TaskText:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return TaskText(payload["features"], payload["padding"])


class TaskTextEncoder(FrozenTextEncoder):
    """The frozen text encoder with the DAgger tensor contract."""

    def __init__(self, max_length: int = 48) -> None:
        super().__init__(max_length=max_length)

    def token_length(self, texts: Sequence[str]) -> int:
        """Tokens needed so that no text in the set is truncated."""
        return max((len(self.tokenizer(text)["input_ids"]) for text in texts), default=1)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> TaskText:
        features, attention = self.encode_tokens(texts, fixed_length=True)
        return TaskText(features, ~attention)
