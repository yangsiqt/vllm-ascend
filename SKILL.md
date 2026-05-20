# SKILL.md - Task1 MTP Support Validation

## Task Overview

- Competition: CCF vLLM Ascend
- Team: 前进4
- Task: Validate and adapt MTP speculative decoding support for MiMo and ERNIE on Ascend NPU.
- Target code base: vLLM Ascend main, commit `64b05b4823ca152741a7aed2d27c77158d432bd7`
- Runtime baseline: Python 3.11, PyTorch 2.10.0, torch-npu 2.10.0, CANN 9.0.0, vLLM `0.20.2rc1.dev367+g0d4d334ea`

## AI Tool Usage

AI tools were used to inspect the vLLM and vLLM Ascend speculative decoding paths, compare the old Task1 implementation with the updated main branch, identify remaining Ascend NPU incompatibilities, generate minimal compatibility patches, and run validation commands.

Main findings:

- New vLLM already recognizes `mimo_mtp` and `ernie_mtp` as MTP model types and normalizes them to `method="mtp"`.
- New vLLM Ascend already routes `method="mtp"` to the Ascend speculative decoding proposer.
- The remaining model-specific issue is in the MTP draft layer implementation: boolean indexing assignment lowers to `NonZero` on Ascend NPU and is not safe during graph capture.
- ERNIE on Ascend 910B also needs compatibility fallbacks for custom operators that are not available in the local CANN 9.0.0 binary set.

## Code Changes

### `vllm_ascend/patch/worker/patch_ernie_mtp.py`

Adds an Ascend-specific monkey patch for `ErnieMultiTokenPredictorLayer`.

The original upstream ERNIE MTP layer uses:

```python
inputs_embeds[positions == 0] = 0
```

On Ascend graph mode this can fail through `aclnnNonzeroV2`. The patch replaces it with:

```python
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```

The rest of the ERNIE MTP forward logic is preserved.

### `vllm_ascend/patch/worker/patch_mimo_mtp.py`

Adds the same Ascend-safe masking change for `MiMoMultiTokenPredictorLayer`.

### `vllm_ascend/patch/worker/__init__.py`

Registers the new ERNIE and MiMo MTP patches during vLLM Ascend plugin initialization.

### `vllm_ascend/ops/layernorm.py`

Avoids `npu_add_rms_norm_bias` on Ascend A2/910B, where CANN 9.0.0 reports no binary support for `AddRmsNormBias`. The fallback uses `torch_npu.npu_add_rms_norm` and applies norm bias separately when needed.

### `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`

Skips AddRmsNormBias-dependent norm-quant fusion patterns when the current Ascend target does not support `npu_add_rms_norm_bias`.

### `vllm_ascend/device/device_op.py`

Keeps custom MoE fast paths on supported chips, while using 910B-compatible fallbacks for:

- `MoeInitRoutingCustom` -> `torch_npu.npu_moe_init_routing_v2`
- `MoeGatingTopK` with `renorm=1` -> fused top-k with explicit renormalization

## Validation

Validation was performed on Ascend 910B with CANN 9.0.0 enabled by:

```bash
source /root/envs/vllm-ascend-2.10/bin/activate
source /data/Ascend/cann-9.0.0/set_env.sh
```

Smoke tests completed before preparing this branch:

| Case | Result | Notes |
| --- | --- | --- |
| ERNIE non-MTP graph | PASS | Baseline graph path can generate text. |
| ERNIE MTP n=1 graph | PASS after patch | `ErnieMTPModel` loaded and ACL graph replayed. |
| MiMo MTP n=1 graph | PASS after patch | `MiMoMTPModel` loaded and ACL graph replayed. |

Important graph-mode setting:

```python
compilation_config={
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [2],
}
```

For `num_speculative_tokens=1`, graph capture size must be aligned to `num_speculative_tokens + 1`, so `[2]` is required.

Representative logs from local validation:

- `/data/task1_new_ernie_nonmtp_graph_moe_fix_smoke.log`
- `/data/task1_new_ernie_mtp_n1_graph_optimized_smoke.log`
- `/data/task1_new_mimo_mtp_n1_graph_smoke.log`

## Performance Notes

All performance checks below used `enforce_eager=False`, `method="mtp"` for MTP cases, 4 prompts, and `max_tokens=256`.

| Model | Mode | max_num_batched_tokens | output_tps | Log |
| --- | --- | ---: | ---: | --- |
| ERNIE-4.5-21B-A3B-PT | non-MTP graph | 4096 | 195.0653 | `/data/task1_new_perf_ernie_nonmtp_graph_4p256_bt4096.log` |
| ERNIE-4.5-21B-A3B-PT | MTP n=1 graph | 4096 | 216.1387 | `/data/task1_new_perf_ernie_mtp_n1_graph_4p256_bt4096.log` |
| MiMo-7B-Base | non-MTP graph | 2048 | 276.1622 | `/data/task1_new_perf_mimo_nonmtp_graph_4p256.log` |
| MiMo-7B-Base | MTP n=1 graph | 2048 | 351.7899 | `/data/task1_new_perf_mimo_mtp_n1_graph_4p256.log` |

For ERNIE MTP, `max_num_batched_tokens=2048` was too small for the tested 4-prompt speculative decoding workload and reduced throughput to `98.7729` output tokens/s. Increasing it to `4096` restored ACL graph efficiency and made MTP faster than the non-MTP baseline.

## Scope

This branch keeps the Task1 change focused on MTP graph compatibility and Ascend 910B operator availability. It does not port the old Task1 branch's broad async scheduling changes, because the updated vLLM Ascend main already contains a newer MTP framework and broad scheduling changes would increase risk.
