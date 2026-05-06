# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for standard decoder-only transformer model files."""

from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from typing import Any, TypeAlias

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .interfaces import EagleModelMixin
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)

StackedParamMapping: TypeAlias = tuple[str, str, str | int]

DEFAULT_STACKED_PARAMS_MAPPING: tuple[StackedParamMapping, ...] = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


class StandardDecoderLayerMixin:
    """Residual + attention + MLP forward path shared by decoder layers."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class StandardDecoderModel(nn.Module, EagleModelMixin):
    """Pipeline-parallel transformer decoder backbone.

    Model files provide architecture-specific layers and config handling while
    this base owns the common embedding, layer iteration, final norm, and stacked
    weight loading behavior.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_factory: Callable[[str], nn.Module],
        prefix: str = "",
        embed_tokens_prefix: str | None = None,
        load_weights_ignore_missing: bool = False,
        named_parameters_remove_duplicate: bool = True,
        stacked_params_mapping: Sequence[StackedParamMapping] = (
            DEFAULT_STACKED_PARAMS_MAPPING
        ),
    ) -> None:
        super().__init__()

        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.load_weights_ignore_missing = load_weights_ignore_missing
        self.named_parameters_remove_duplicate = named_parameters_remove_duplicate
        self.stacked_params_mapping = tuple(stacked_params_mapping)

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            embed_tokens_kwargs: dict[str, Any] = {
                "quant_config": quant_config,
            }
            if embed_tokens_prefix is not None:
                embed_tokens_kwargs["prefix"] = embed_tokens_prefix
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                **embed_tokens_kwargs,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            layer_factory,
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual, **extra_layer_kwargs
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _load_stacked_weight(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
    ) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if weight_loader == default_weight_loader:
            weight_loader(param, loaded_weight)
        else:
            weight_loader(param, loaded_weight, shard_id)

    def _load_unstacked_weight(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(
            self.named_parameters(
                remove_duplicate=self.named_parameters_remove_duplicate
            )
        )
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            loaded = False
            skip = False
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    skip = True
                    break
                if is_pp_missing_parameter(mapped_name, self):
                    skip = True
                    break

                param = params_dict[mapped_name]
                self._load_stacked_weight(param, loaded_weight, shard_id)
                name = mapped_name
                loaded = True
                break

            if skip:
                continue

            if not loaded:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    if self.load_weights_ignore_missing:
                        continue
                    raise KeyError(name)

                param = params_dict[name]
                self._load_unstacked_weight(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class StandardCausalLMMixin:
    """Shared pass-through methods for decoder-only causal LM wrappers."""

    config: Any
    lm_head: nn.Module
    logits_processor: LogitsProcessor
    model: StandardDecoderModel

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
