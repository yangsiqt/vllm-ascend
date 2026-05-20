# AI Development Record: use_prefill_query_quantization

## Task

Adapt vLLM Ascend for:

```bash
--attention-config '{"use_prefill_query_quantization": true}'
```

The goal is to enable Query quantization in supported Ascend prefill attention paths while keeping unsupported paths compatible through warning fallback.

## Implementation Summary

- Added attention-config reading for `use_prefill_query_quantization`.
- Added an experimental dense `PrefillNoCache` quantized prefill path:
  - Q/K/V are dynamically quantized to int8.
  - The path pads vLLM's TND prefill tensors to BSND, calls `torch_npu.npu_prompt_flash_attention` with int8
    inputs and dequant/quant scales, then flattens the result back to TND.
  - A TND `npu_fused_infer_attention_score` attempt was tested but rejected by CANN because all-int8 Q/K/V is not
    supported in TND layout.
- Added safe handling for MLA prefill query quantization:
  - The first direct FIA v2 query-dequant attempt was validated on Ascend 910B2 and rejected by CANN with
    `PFA not support query dequant now`.
  - The final implementation therefore warns once and falls back to the existing prefill PFA path instead of
    crashing.
- Kept the default behavior unchanged when the parameter is `False`.
- Added safe fallback warnings for unsupported dense attention states and C8 prefill paths.

## Supported Scope

- Supported: dense attention `PrefillNoCache` path through `npu_prompt_flash_attention` int8 Q/K/V.
- Fallback: MLA prefill PFA, MLA chunked prefill context, dense cache-hit/chunked prefill, and C8 attention paths.
- Scope for the first PR: eager-mode validation. Graph / ACLGraph mode is not claimed as fully supported.

## AI Assistance

AI was used to inspect vLLM upstream attention config behavior, locate Ascend attention backends, identify compatible `torch_npu` operator interfaces, implement the patch, and prepare validation notes.

## Validation Plan

```bash
ruff check vllm_ascend/attention/mla_v1.py vllm_ascend/attention/attention_v1.py tests/ut/attention/test_mla_v1.py tests/ut/attention/test_attention_v1.py
ruff format --check vllm_ascend/attention/mla_v1.py vllm_ascend/attention/attention_v1.py tests/ut/attention/test_mla_v1.py tests/ut/attention/test_attention_v1.py
python -m py_compile vllm_ascend/attention/mla_v1.py vllm_ascend/attention/attention_v1.py
pytest tests/ut/attention/test_mla_v1.py tests/ut/attention/test_attention_v1.py
git diff --check
```

For real performance data, use an Ascend quantized MLA model and compare long-prompt prefill latency with the parameter disabled and enabled.
For the dense experimental path, also compare against a dense model such as MiMo/Qwen-style attention with long prompts.

## Validation Result

- Rebased the Task2 branch onto latest `vllm-project/vllm-ascend` main:
  - vllm-ascend commit: `b9770bc6143f1c1c19f8de31af61aa1e6e2077de`
  - required CANN: `9.0.0`
  - required PyTorch / torch-npu: `2.10.0 / 2.10.0`
  - required vLLM main commit: `0d4d334eaa583b9c09aa4eb7538c22db99fd84b3`
- Local source trees were aligned to those commits, but the current server runtime is still CANN `8.5.1` and
  torch-npu `2.9.0rc1`; final on-device performance validation must be rerun in the CANN 9.0 / torch-npu 2.10
  image.
- Downloaded and validated `/data/models/DeepSeek-V2-Lite-w8a8`.
- `use_prefill_query_quantization=False` passed model load and generation.
- `use_prefill_query_quantization=True` passed model load and generation through fallback.
- Direct query-dequant FIA v2 attempts were rejected by CANN on Ascend 910B2 with `PFA not support query dequant now`,
  so this PR does not claim accelerated query-quant prefill performance on this hardware/runtime.
- Additional `kv_cache_dtype=fp8_e4m3` checks failed for both enabled and disabled query quantization before
  generation in `npu_kv_rmsnorm_rope_cache`, confirming that the upstream-style FP8 KV-cache prerequisite is
  not currently usable for this MLA model on the tested Ascend stack.
- Added mock unit coverage for the dense `PrefillNoCache` int8 PFA path.
- After NPU recovery, PFA int8 accuracy probes passed:
  - relative MAE was about 1.86% at 128 tokens, 2.22% at 512 tokens, and 2.47% at 1024 tokens with
    `softmax_quant_scale=127`.
- MiMo-7B-Base smoke passed with `use_prefill_query_quantization=True` and `enable_prefix_caching=False`.
- MiMo-7B-Base long-prompt performance did not improve:
  - 1024/32: 0.5461 s disabled vs 0.5743 s enabled.
  - 2048/32: 0.6265 s disabled vs 0.6631 s enabled.
  - 4096/32: 0.7515 s disabled vs 0.7918 s enabled.
- A `torch_npu.npu_quantize` replacement was also tried, but 4096/32 regressed further to 0.8385 s enabled, so
  the manual quantization path was kept.
- Current conclusion: functional support exists for the guarded dense prefill path, but the dynamic quantization
  and layout conversion overhead outweighs the int8 PFA benefit in the tested MiMo scenarios.
- The MiMo performance numbers above were collected on the old CANN 8.5.1 / torch-npu 2.9 runtime and are now
  treated as exploratory, not final submission data for the latest main branch.
