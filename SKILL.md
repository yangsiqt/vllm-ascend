# SKILL.md — 前进4 团队开发记录

## 任务概述

- **比赛**: CCF开源创新大赛 — vLLM Ascend
- **团队**: 前进4
- **任务**: `--speculative-config` 参数 → `method` 子参数 → `mimo_mtp`、`ernie_mtp` 的 Ascend NPU 适配与验证

## AI 工具使用方式

本项目使用 **Claude Code** (通过 VS Code 扩展) 进行主要 AI 辅助开发，并使用
**Codex** 进行最终代码 review、lint 修复和最终复测。关键使用模式:

### 1. 代码分析与理解

- 使用 Claude Code 阅读上游 vLLM 源码 (`vllm/config/speculative.py`, `vllm/model_executor/models/mimo_mtp.py`, `ernie_mtp.py`)
- 理解 MTP (Multi-Token Prediction) speculative decoding 机制: 17种 MTP 模型类型在 `hf_config_override()` 中映射，最终所有特定 MTP 类型被规范化为 `method="mtp"`
- 理解 vllm-ascend 的 monkey-patch 模式: 通过替换上游类实现 NPU 适配

### 2. 代码生成

- 参考已有 patch 文件 (`patch_deepseek_mtp.py`, `patch_qwen3_next_mtp.py`) 的模式
- 生成 `patch_mimo_mtp.py` 和 `patch_ernie_mtp.py`
- AI 负责:
    - 分析上游类的 `forward` 方法
    - 识别 NPU 不安全的操作 (in-place 索引赋值)
    - 替换为 `torch.where` (NPU 安全的 masked fill)
    - 保持 monkey-patch 模式一致

### 3. 调试与修复

- Python 环境问题: `python` vs `/usr/local/python3.11.14/bin/python` 路径
- vLLM 版本兼容: 从 0.20.2 切换到 0.19.2rc0 (匹配 vllm-ascend 0.19.1rc2)
- numpy/numba 冲突: 降级 numpy 以兼容 numba
- HF 模型下载失败: 尝试多种镜像和认证方式

### 4. 验证

- Import 链验证: 确认所有模块可导入
- Monkey-patch 验证: 确认类替换正确生效
- 引擎初始化验证: 使用 `--load-format dummy` 验证到 LLM 引擎初始化阶段
- 端到端推理验证: `llm.generate`、`vllm serve`、non-MTP baseline、MTP n=1/n=2
- ERNIE-4.5-21B-A3B-PT 验证: non-MTP 和 MTP n=1 均完成端到端生成，并完成 4 prompt x 256 tokens 性能对比
- 代码质量检查: `ruff check`、`ruff format --check`、`py_compile`、`git diff --check`
- 最终性能复测: 使用最终代码运行 `/data/bench_final.py`

## 代码改动说明

### 新增文件

#### 1. `vllm_ascend/patch/worker/patch_mimo_mtp.py`

- 创建 `AscendMiMoMultiTokenPredictorLayer`，继承自上游 `MiMoMultiTokenPredictorLayer`
- 关键改动: 将 `inputs_embeds[(positions == 0)] = 0` 替换为 `inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)`
    - 原因: NPU 上 in-place 索引赋值可能导致显存访问错误
    - `torch.where` 是融合算子，更安全高效
- Monkey-patch: `vllm.model_executor.models.mimo_mtp.MiMoMultiTokenPredictorLayer = AscendMiMoMultiTokenPredictorLayer`

#### 2. `vllm_ascend/patch/worker/patch_ernie_mtp.py`

- 创建 `AscendErnieMultiTokenPredictorLayer`，继承自上游 `ErnieMultiTokenPredictorLayer`
- 同样的 `torch.where` 替换模式
- 使用 Ernie 特定的 norm/proj 命名 (`mtp_emb_norm`, `mtp_hidden_norm`, `mtp_linear_proj`)

### 修改文件

#### 3. `vllm_ascend/patch/worker/__init__.py`

- 新增两行 import 注册 patch:
  ```python
  import vllm_ascend.patch.worker.patch_mimo_mtp  # noqa
  import vllm_ascend.patch.worker.patch_ernie_mtp  # noqa
  ```

#### 4. `vllm_ascend/spec_decode/__init__.py`

- 修改第37行，将 `mimo_mtp` 和 `ernie_mtp` 添加到 EagleProposer 路由:
  ```python
  # Before: elif method in ("eagle", "eagle3", "mtp"):
  # After:  elif method in ("eagle", "eagle3", "mtp", "mimo_mtp", "ernie_mtp"):
  ```
- 原因: 上游 vLLM 将所有特定 MTP 类型规范化为 `method="mtp"`，但在到达 `spec_decode` 路由之前，`mimo_mtp`/`ernie_mtp` 仍以原始名称传递。`get_spec_decode_method()` 需要识别这些原始名称并路由到 `AscendEagleProposer`。

#### 5. `vllm_ascend/device/device_op.py`

- ERNIE-4.5 MoE 在 Ascend 910B 上会触发 `MoeGatingTopK renorm=1` 不支持。
- 处理方式: 算子侧使用 `renorm=0`，随后对 top-k 权重手动归一化，保持语义一致。
- `MoeInitRoutingCustom` 在当前 910B 环境中没有可用 binary，改用 `torch_npu.npu_moe_init_routing_v2`。

#### 6. `vllm_ascend/ops/layernorm.py`

- ERNIE 默认路径会调用 `AddRmsNormBias` 自定义算子，但当前 910B CANN 环境不支持该 op。
- 增加受控 fallback: 仅当错误明确为 `AddRmsNormBias` 不支持当前 SoC 时，退回 `torch_npu.npu_add_rms_norm`，再手动加 bias。
- 该修复使 ERNIE non-MTP 和 ERNIE MTP n=1 均可在不设置 `VLLM_BATCH_INVARIANT=1` 的默认路径下运行。

## 技术要点

### MTP 架构理解

1. `vllm/config/speculative.py:303` 的 `hf_config_override()` 根据 target model architecture 映射到 MTP 类型
   - `MiMoForCausalLM` → `mimo_mtp`
   - `ernie4_5_moe` → `ernie_mtp`
2. `speculative.py:512` 将所有特定 MTP 类型规范化为 `method="mtp"`
3. MTP draft model 复用 target model 的 embedding 和 lm_head
4. MTP 使用 `ParallelLMHead` 进行多头预测

### NPU 适配要点

- **禁止** in-place 索引赋值 (`tensor[indices] = value`)
- **使用** `torch.where` 进行 masked fill
- **使用** `assert inputs_embeds is not None` 避免 None 检查导致的图断裂
- 保持与上游类相同的 forward 签名

## Commit 2: async_scheduling NPU 修复

commit `920705c88`

MTP + `async_scheduling=True` (V1 默认) 在 Ascend NPU 上 hang，根因是三个 NPU 不兼容：

### 2.1 H2D 异步拷贝 hang
`pin_memory().to(device, non_blocking=True)` 在 NPU 上 hang，改为 `non_blocking=False`（7 处）。

### 2.2 torch.npu.Event().record() hang
新创建的 `Event.record()` 在 NPU 上 hang。替代方案：
- 阻塞拷贝（`non_blocking=False`）隐式同步 → 无需 event
- `torch.npu.synchronize()` 替代 event wait

### 2.3 copy_ 源/目标重叠
`tensor.copy_(torch.where(condition, values, tensor))` 在 NPU 上源和目标不能重叠。拆为中间张量。

### 修改文件
- `eagle_proposer.py`: H2D 拷贝 `non_blocking=False`，Triton kernel 后加 sync
- `model_runner_v1.py`: 7 处 `non_blocking=False`，event→sync，seq_lens 阻塞拷贝，attention metadata 修复
- `utils.py`: copy_ 中间张量

## Commit 3: 冗余 sync 点移除

commit `214515e98`

Commit 2 通过阻塞拷贝和 sync 机制保证数据同步后，5 个防御性 `torch.npu.synchronize()` 变为冗余：

| 位置 | 原因 |
|------|------|
| eagle_proposer: draft forward 后 | 阻塞拷贝已保证数据就绪 |
| model_runner: prepare_inputs 前 | 同上 |
| model_runner: forward 前 | NPU 上此前操作均为同步 |
| model_runner: MTP unpack 前 | forward 是同步的 |
| model_runner: AsyncOutput 前 | 阻塞拷贝隐式同步 |

## 验证结果

### 功能验证
| 验证项 | 状态 |
|--------|------|
| import 链 | ✅ |
| 引擎初始化 | ✅ |
| MTP n=1 推理 | ✅ |
| MTP n=2 推理 | ✅ (n=2 比 n=1 慢, 见性能数据) |
| 离线 API (llm.generate) | ✅ |
| 在线 API (server) | ✅ |
| Non-MTP baseline | ✅ |
| temperature=0 确定性 | ✅ |

### 性能数据 (MiMo-7B-Base, Ascend NPU)

最终复测日期: 2026-05-16。测试命令: `/data/bench_final.py`。

| 场景 | non-MTP | MTP n=1 | MTP n=2 | n=1 加速 | n=2 加速 | 一致性 |
|------|--------:|--------:|--------:|---------:|---------:|:------:|
| 1p × 64t | 65 tok/s | 91 tok/s | 84 tok/s | **+39.7%** | +27.9% | 100% |
| 1p × 256t | 65 tok/s | 94 tok/s | 90 tok/s | **+44.8%** | +38.1% | 100% |
| 1p × 512t | 65 tok/s | 100 tok/s | 96 tok/s | **+53.0%** | +47.2% | 100% |
| 1p × 1024t | 66 tok/s | 101 tok/s | 98 tok/s | **+54.1%** | +49.6% | 100% |
| 2p × 256t | 124 tok/s | 185 tok/s | 175 tok/s | **+49.6%** | +41.3% | 100% |
| 2p × 1024t | 125 tok/s | 200 tok/s | 192 tok/s | **+60.2%** | +53.8% | 100% |
| 4p × 256t | 248 tok/s | 351 tok/s | 346 tok/s | **+41.6%** | +39.5% | 100% |
| 4p × 512t | 240 tok/s | 367 tok/s | 362 tok/s | **+52.9%** | +50.8% | 100% |

### MTP n=2 vs n=1

n=2 第二个 draft token 接受率较低，导致多数场景总吞吐低于 n=1。MiMo-7B
推荐配置为 `num_speculative_tokens=1`。

### ERNIE-4.5-21B-A3B-PT 补充验证

验证环境: Ascend 910B 64GB 单卡，`enforce_eager=True`，不设置 `VLLM_BATCH_INVARIANT=1`。

| 验证项 | 状态 |
|--------|------|
| ERNIE non-MTP 生成 | ✅ |
| ERNIE MTP n=1 生成 | ✅ |
| target 架构识别为 `Ernie4_5_MoeForCausalLM` | ✅ |
| draft 架构识别为 `ErnieMTPModel` | ✅ |
| MTP embedding/lm_head 共享日志 | ✅ |

4 prompt x 256 tokens 性能对比:

| 场景 | output_tps | total_tps | 说明 |
|------|-----------:|----------:|------|
| ERNIE non-MTP | 55.0610 | 57.8033 | 1024 output tokens |
| ERNIE MTP n=1 | 79.1647 | 83.1074 | 1024 output tokens |

ERNIE MTP n=1 相比 non-MTP 的输出吞吐提升为 **+43.8%**。由于 ERNIE-4.5-21B-A3B-PT 配置中 `num_nextn_predict_layers=1`，推荐验证配置为 `num_speculative_tokens=1`。

### 最终代码质量检查

Codex 最终 review 后补充了 lint 收尾修复，并完成以下检查:

- `ruff check`: 通过
- `ruff format --check`: 通过
- `py_compile`: 通过
- `git diff --check`: 通过
- MTP + async + `logprobs=1`: 通过

### 外部修复

vLLM upstream 的 async output/logprobs 兼容性修复建议单独提交到 vLLM。比赛主
交付聚焦 vllm-ascend 仓库中的 MTP 适配代码与 `SKILL.md`。

## 限制

- MiMo-7B-Base 模型通过 ModelScope 下载 (14.9GB, 4分片)
- ERNIE-4.5-21B-A3B-PT 已完成 non-MTP 与 MTP n=1 验证；n=2 不作为推荐配置，因为该模型配置中 `num_nextn_predict_layers=1`
- `repetition_penalty` / `presence_penalty` 等 async output 相关场景建议配合 vLLM upstream 兼容性修复使用
