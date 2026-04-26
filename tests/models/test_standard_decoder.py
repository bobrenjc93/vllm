# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.models.standard_decoder import (
    DEFAULT_STACKED_PARAMS_MAPPING,
    StandardDecoderLayerMixin,
    StandardDecoderModel,
)

pytestmark = pytest.mark.skip_global_cleanup


class _FakeNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states + 1
        return hidden_states + 1, residual + 2


class _FakeAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states + positions.unsqueeze(-1)


class _FakeMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 3


class _FakeDecoderLayer(StandardDecoderLayerMixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _FakeNorm()
        self.self_attn = _FakeAttention()
        self.post_attention_layernorm = _FakeNorm()
        self.mlp = _FakeMLP()


class _ParamModule(nn.Module):
    def __init__(
        self,
        name: str,
        calls: list[tuple[str, torch.Tensor, str | int | None]],
    ) -> None:
        super().__init__()
        param = nn.Parameter(torch.zeros(1))

        def weight_loader(
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: str | int | None = None,
        ) -> None:
            calls.append((name, loaded_weight.clone(), shard_id))
            param.data.copy_(loaded_weight)

        param.weight_loader = weight_loader
        self.register_parameter("weight", param)


class _ScaleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_scale = nn.Parameter(torch.zeros(()))


class _AttentionModule(nn.Module):
    def __init__(
        self,
        calls: list[tuple[str, torch.Tensor, str | int | None]],
    ) -> None:
        super().__init__()
        self.qkv_proj = _ParamModule("qkv_proj", calls)
        self.attn = _ScaleModule()


class _LayerModule(nn.Module):
    def __init__(
        self,
        calls: list[tuple[str, torch.Tensor, str | int | None]],
    ) -> None:
        super().__init__()
        self.self_attn = _AttentionModule(calls)


class _FakeLoaderModel(StandardDecoderModel):
    def __init__(self, *, ignore_missing: bool) -> None:
        nn.Module.__init__(self)
        self.quant_config = None
        self.load_weights_ignore_missing = ignore_missing
        self.named_parameters_remove_duplicate = False
        self.stacked_params_mapping = DEFAULT_STACKED_PARAMS_MAPPING
        self.calls: list[tuple[str, torch.Tensor, str | int | None]] = []
        self.layers = nn.ModuleList([_LayerModule(self.calls)])


def test_standard_decoder_layer_mixin_forward() -> None:
    layer = _FakeDecoderLayer()
    positions = torch.tensor([5.0])
    hidden_states = torch.tensor([[2.0]])

    output, residual = layer(positions, hidden_states, None)

    assert torch.equal(residual, torch.tensor([[4.0]]))
    assert torch.equal(output, torch.tensor([[27.0]]))


def test_standard_decoder_model_loads_stacked_weights() -> None:
    model = _FakeLoaderModel(ignore_missing=False)
    loaded_weight = torch.tensor([7.0])

    loaded_params = model.load_weights(
        [("layers.0.self_attn.q_proj.weight", loaded_weight)]
    )

    assert loaded_params == {"layers.0.self_attn.qkv_proj.weight"}
    assert model.calls == [("qkv_proj", loaded_weight, "q")]


def test_standard_decoder_model_remaps_kv_scales() -> None:
    model = _FakeLoaderModel(ignore_missing=False)

    loaded_params = model.load_weights(
        [("layers.0.self_attn.k_proj.k_scale", torch.tensor(3.0))]
    )

    assert loaded_params == {"layers.0.self_attn.attn.k_scale"}
    assert torch.equal(model.layers[0].self_attn.attn.k_scale, torch.tensor(3.0))


def test_standard_decoder_model_can_ignore_missing_weights() -> None:
    model = _FakeLoaderModel(ignore_missing=True)

    loaded_params = model.load_weights([("missing.weight", torch.tensor([1.0]))])

    assert loaded_params == set()


def test_standard_decoder_model_is_strict_by_default() -> None:
    model = _FakeLoaderModel(ignore_missing=False)

    with pytest.raises(KeyError, match="missing.weight"):
        model.load_weights([("missing.weight", torch.tensor([1.0]))])
