from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pad_sequences(
    sequences: Sequence[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot pad an empty sequence list.")
    lengths = [int(sequence.numel()) for sequence in sequences]
    if any(length < 1 for length in lengths):
        raise ValueError("Prompt token sequences must not be empty.")
    width = max(lengths)
    input_ids = torch.full(
        (len(sequences), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        length = lengths[index]
        input_ids[index, :length] = sequence
        attention_mask[index, :length] = 1
    return input_ids, attention_mask


def center_and_normalize(
    raw_vectors: np.ndarray,
    baseline_vector: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    raw = np.asarray(raw_vectors)
    baseline = np.asarray(baseline_vector)
    if raw.ndim != 2:
        raise ValueError("raw_vectors must be a two-dimensional array.")
    if baseline.ndim != 1:
        raise ValueError("baseline_vector must be a one-dimensional array.")
    if raw.shape[1] != baseline.shape[0]:
        raise ValueError("Baseline width does not match vector width.")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be a positive finite number.")
    if not np.isfinite(raw).all() or not np.isfinite(baseline).all():
        raise ValueError("Non-finite raw or baseline vector encountered.")

    centered = raw.astype(np.float32, copy=True)
    centered -= baseline.astype(np.float32, copy=False)
    norms = np.linalg.norm(centered, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Non-finite centered-vector norm encountered.")
    if np.any(norms <= float(epsilon)):
        raise ValueError("Centered-vector norm is not greater than epsilon.")
    normalized = centered / norms[:, None]
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized vector encountered.")
    return normalized.astype(np.float32, copy=False)


class ChemDFMContextEmbeddingBackend:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "mps",
        dtype: str = "float16",
    ) -> None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        self.hidden_size = int(self.model.config.hidden_size)

    def encode_prompts(
        self,
        *,
        system_prompt: str,
        user_prompts: Sequence[str],
        batch_size: int,
    ) -> np.ndarray:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not user_prompts:
            return np.empty((0, self.hidden_size), dtype=np.float32)

        base_model = getattr(self.model, "model", None)
        if base_model is None:
            raise RuntimeError("ChemDFM base transformer is unavailable.")

        batches: list[np.ndarray] = []
        for start in range(0, len(user_prompts), int(batch_size)):
            prompts = user_prompts[start : start + int(batch_size)]
            sequences = [
                torch.tensor(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                    ),
                    dtype=torch.long,
                )
                for user_prompt in prompts
            ]
            input_ids, attention_mask = _pad_sequences(
                sequences,
                self.tokenizer.pad_token_id,
            )
            device_ids = input_ids.to(self.device)
            device_mask = attention_mask.to(self.device)
            with torch.inference_mode():
                output = base_model(
                    input_ids=device_ids,
                    attention_mask=device_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.last_hidden_state
                if hidden.ndim != 3 or hidden.shape[:2] != device_mask.shape:
                    raise RuntimeError("Unexpected final hidden-state shape.")
                last_indices = device_mask.sum(dim=1).to(torch.long) - 1
                if bool(torch.any(last_indices < 0)):
                    raise RuntimeError("Prompt contains no non-padding token.")
                batch_indices = torch.arange(
                    hidden.shape[0], device=hidden.device
                )
                vectors = hidden[batch_indices, last_indices].float()
            if vectors.shape[1] != self.hidden_size:
                raise RuntimeError("Hidden width differs from model configuration.")
            if not bool(torch.isfinite(vectors).all()):
                raise ValueError("Non-finite context embedding encountered.")
            batches.append(vectors.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0)
