from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import semantic_fingerprint.screening.backend as backend_module
from semantic_fingerprint.screening.backend import (
    ChemDFMContextEmbeddingBackend,
    center_and_normalize,
)


class PromptTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.calls.append((messages, tokenize, add_generation_prompt))
        assert messages[0] == {"role": "system", "content": "system"}
        assert tokenize is True
        assert add_generation_prompt is True
        return {
            "short": [10, 11],
            "long": [20, 21, 22, 23],
            "middle": [30, 31, 32],
        }[messages[1]["content"]]


class TrackingBaseModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.grad_enabled: list[bool] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        self.grad_enabled.append(torch.is_grad_enabled())
        self.calls.append((input_ids.cpu().clone(), attention_mask.cpu().clone()))
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.float16
        ).expand(input_ids.shape[0], -1)
        hidden = torch.stack(
            (
                input_ids.to(torch.float16),
                positions,
                torch.ones_like(positions),
            ),
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


class NoHeadModel:
    def __init__(self) -> None:
        self.model = TrackingBaseModel()
        self.generate_called = False
        self.full_forward_called = False
        self.config = SimpleNamespace(hidden_size=3, use_cache=False)

    def __call__(self, **kwargs: object) -> None:
        self.full_forward_called = True
        raise AssertionError("The causal language-model head must not run.")

    def generate(self, **kwargs: object) -> None:
        self.generate_called = True
        raise AssertionError("Generation must not run.")


def _fake_backend() -> ChemDFMContextEmbeddingBackend:
    backend = object.__new__(ChemDFMContextEmbeddingBackend)
    backend.device = torch.device("cpu")
    backend.tokenizer = PromptTokenizer()
    backend.model = NoHeadModel()
    backend.hidden_size = 3
    return backend


def test_encode_prompts_uses_last_nonpadding_final_layer_without_normalizing() -> None:
    backend = _fake_backend()

    vectors = backend.encode_prompts(
        system_prompt="system",
        user_prompts=["short", "long", "middle"],
        batch_size=2,
    )

    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 3)
    np.testing.assert_array_equal(
        vectors,
        np.array(
            [
                [11.0, 1.0, 1.0],
                [23.0, 3.0, 1.0],
                [32.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert not np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert backend.model.full_forward_called is False
    assert backend.model.generate_called is False
    assert backend.model.model.grad_enabled == [False, False]

    first_ids, first_mask = backend.model.model.calls[0]
    assert first_ids.tolist() == [[10, 11, 0, 0], [20, 21, 22, 23]]
    assert first_mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    second_ids, second_mask = backend.model.model.calls[1]
    assert second_ids.tolist() == [[30, 31, 32]]
    assert second_mask.tolist() == [[1, 1, 1]]

    assert [call[0][1]["content"] for call in backend.tokenizer.calls] == [
        "short",
        "long",
        "middle",
    ]


@pytest.mark.parametrize("bad_batch_size", [0, -1, True, 1.5])
def test_encode_prompts_rejects_invalid_batch_size(bad_batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _fake_backend().encode_prompts(
            system_prompt="system",
            user_prompts=["short"],
            batch_size=bad_batch_size,  # type: ignore[arg-type]
        )


def test_encode_prompts_returns_hidden_width_for_empty_input() -> None:
    result = _fake_backend().encode_prompts(
        system_prompt="system",
        user_prompts=[],
        batch_size=2,
    )

    assert result.shape == (0, 3)
    assert result.dtype == np.float32


class InitModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(hidden_size=5120, use_cache=True)
        self.device_argument: torch.device | None = None
        self.eval_called = False
        self.model = object()

    def to(self, device: torch.device) -> None:
        self.device_argument = device

    def eval(self) -> None:
        self.eval_called = True


def test_backend_loads_local_model_in_eval_mode_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(
        pad_token_id=None,
        pad_token=None,
        eos_token="<eos>",
    )
    model = InitModel()
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    def load_tokenizer(path: str, **kwargs: object) -> object:
        tokenizer_calls.append((path, kwargs))
        return tokenizer

    def load_model(path: str, **kwargs: object) -> InitModel:
        model_calls.append((path, kwargs))
        return model

    monkeypatch.setattr(
        backend_module.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        backend_module.AutoModelForCausalLM,
        "from_pretrained",
        load_model,
    )

    backend = ChemDFMContextEmbeddingBackend(
        "/models/tiny",
        device="cpu",
        dtype="float32",
    )

    assert backend.tokenizer is tokenizer
    assert backend.model is model
    assert backend.hidden_size == 5120
    assert tokenizer_calls == [("/models/tiny", {"local_files_only": True})]
    assert model_calls == [
        (
            "/models/tiny",
            {"torch_dtype": torch.float32, "local_files_only": True},
        )
    ]
    assert tokenizer.pad_token == "<eos>"
    assert model.device_argument == torch.device("cpu")
    assert model.eval_called is True
    assert model.config.use_cache is False


def test_backend_rejects_unsupported_dtype_unavailable_mps_and_missing_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        ChemDFMContextEmbeddingBackend(
            "/models/tiny", device="cpu", dtype="bfloat16"
        )

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        ChemDFMContextEmbeddingBackend(
            "/models/tiny", device="mps", dtype="float16"
        )

    backend = _fake_backend()
    backend.model.model = None
    with pytest.raises(RuntimeError, match="base transformer"):
        backend.encode_prompts(
            system_prompt="system",
            user_prompts=["short"],
            batch_size=1,
        )


def test_center_and_normalize_subtracts_baseline_and_returns_unit_float32() -> None:
    raw = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float16)
    baseline = np.array([1.0, 0.0], dtype=np.float32)

    centered = center_and_normalize(raw, baseline)

    assert centered.dtype == np.float32
    np.testing.assert_allclose(
        centered,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(np.linalg.norm(centered, axis=1), 1.0)
    np.testing.assert_array_equal(raw, [[2.0, 0.0], [1.0, 3.0]])


def test_center_and_normalize_keeps_axis_baselines_isolated() -> None:
    raw = np.array([[2.0, 2.0]], dtype=np.float32)

    axis_a = center_and_normalize(raw, np.array([1.0, 2.0]))
    axis_b = center_and_normalize(raw, np.array([2.0, 1.0]))

    np.testing.assert_allclose(axis_a, [[1.0, 0.0]])
    np.testing.assert_allclose(axis_b, [[0.0, 1.0]])


@pytest.mark.parametrize(
    ("raw", "baseline", "message"),
    [
        (np.array([1.0, 2.0]), np.array([0.0, 0.0]), "two-dimensional"),
        (np.ones((2, 3)), np.ones(2), "width"),
        (np.array([[np.nan, 1.0]]), np.zeros(2), "Non-finite"),
        (np.ones((1, 2)), np.array([np.inf, 0.0]), "Non-finite"),
        (np.ones((1, 2)), np.ones(2), "norm"),
    ],
)
def test_center_and_normalize_rejects_invalid_or_degenerate_vectors(
    raw: np.ndarray,
    baseline: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        center_and_normalize(raw, baseline)


def test_center_and_normalize_honors_strict_epsilon_threshold() -> None:
    raw = np.array([[1.0 + 1e-7, 1.0]], dtype=np.float64)
    baseline = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="norm"):
        center_and_normalize(raw, baseline, epsilon=1e-6)
    result = center_and_normalize(raw, baseline, epsilon=1e-8)
    np.testing.assert_allclose(result, [[1.0, 0.0]])


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf])
def test_center_and_normalize_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        center_and_normalize(
            np.array([[2.0, 0.0]]),
            np.array([1.0, 0.0]),
            epsilon=epsilon,
        )
