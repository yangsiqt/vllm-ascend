# SKILL.md — 前进4 团队开发记录

## 任务概述

- **比赛**: CCF开源创新大赛 — vLLM Ascend
- **团队**: 前进4
- **任务1**: `--speculative-config` 参数 → `method` 子参数 → `mimo_mtp`、`ernie_mtp` 的 Ascend NPU 适配
- **任务2**: `--attention-config` 参数 → `use_prefill_query_quantization` 子参数 (待开发)

## AI 工具使用方式

本项目使用 **Claude Code** (通过 VS Code 扩展) 进行 AI 辅助开发。关键使用模式:

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
- Pre-commit CI: 代码风格和基础检查

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

## 限制与后续

- 完整端到端推理测试需要修复 Triton-Ascend 驱动激活问题 (预置环境问题，非代码问题)
- MiMo-7B-Base 模型已通过 ModelScope 下载完成 (14.9GB, 4分片)
- Ernie4.5-MoE 模型未公开发布，需要华为内部获取
- `--attention-config` 任务 (难度:高) 待开发

## 提交信息

```bash
git commit -s -m "feat(spec_decode): add Ascend NPU support for mimo_mtp and ernie_mtp"

- Add AscendMiMoMultiTokenPredictorLayer and AscendErnieMultiTokenPredictorLayer
  with NPU-safe torch.where masking
- Register patches in patch/worker/__init__.py
- Route mimo_mtp and ernie_mtp to AscendEagleProposer in spec_decode/__init__.py
- Follows existing Ascend patch pattern (cf. patch_deepseek_mtp.py)
```
