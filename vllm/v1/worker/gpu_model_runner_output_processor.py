# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.distributed

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
)
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    NgramProposerGPU,
)
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    num_reqs = len(finished_mask)

    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )

        num_finished = sum(finished_mask)
        if num_finished == 0:
            return [None] * num_reqs
        if num_finished == num_reqs:
            return list(raw_pooler_output.to("cpu", non_blocking=True))

        # partial finished
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            partial_pooler_output[i] = out
        return partial_pooler_output

    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            pooler_output[i] = out.to("cpu", non_blocking=True)
    return pooler_output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=self._raw_pooler_output,
                finished_mask=finished_mask,
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        This function blocks until the copy is finished.
        """
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output
        return self._model_runner_output


class GPUModelRunnerOutputProcessorMixin:
    def _get_or_create_async_output_copy_stream(self) -> torch.cuda.Stream:
        stream = self.async_output_copy_stream
        if stream is None:
            stream = torch.cuda.Stream()
            self.async_output_copy_stream = stream
        return stream

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=self.input_batch.req_ids,
            finished_mask=finished_mask,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if not current_platform.is_cuda_alike():
            # cpu/xpu runners cannot use the CUDA stream/event-based wrapper.
            model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
            )
            self._sync_device()
            return model_runner_output

        return AsyncGPUPoolingModelRunnerOutput(
            model_runner_output=model_runner_output,
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
            async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: SchedulerOutput,
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.use_eagle()
                or spec_config.uses_draft_model()
                or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
            elif (
                spec_config.use_ngram_gpu()
                and not spec_config.disable_padded_drafter_batch
            ):
                assert isinstance(self.drafter, NgramProposerGPU)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count, _ = (
                        self.drafter.update_token_ids_ngram(
                            sampled_token_ids,
                            self.input_batch,
                            self.token_ids_gpu_tensor,
                            self.num_tokens_no_spec_gpu,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

            if not input_fits_in_drafter:
                # Zero out draft tokens so the scheduler doesn't schedule
                # stale drafts from the previous step.
                # For Nemotron-H: it is necessary to zero out the draft tokens,
                # otherwise the stale tokens will corrupt Mamba recurrent
                # state and logprobs for sequences near max_model_len.
                self._draft_token_ids = torch.zeros(
                    1, device=self.device, dtype=torch.int32
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if spec_config is not None:
            self.finalize_kv_connector()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.routed_experts_initialized:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )

        if not self.use_async_scheduling:
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: SchedulerOutput, zeros_only: bool = False
    ) -> None:
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()
