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

- `/data/patched_freshcache_ernie_mtp_n1_graph_cann900_smoke.log`
- `/data/mimo_mtp_n1_graph_cann900_after_patch_smoke.log`

## Scope

This branch intentionally keeps the Task1 change small. It does not port the old Task1 branch's broad async scheduling, MoE fallback, or RMSNorm fallback changes, because the updated vLLM Ascend main already contains a newer MTP framework and those older changes would increase risk and may hurt performance.
