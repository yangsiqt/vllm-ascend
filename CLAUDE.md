# CLAUDE.md — 前进4 团队 · CCF开源创新大赛 vLLM Ascend

## 环境

- **服务器**: Ascend NPU, CentOS, 64GB RAM
- **Python**: `/usr/local/python3.11.14/bin/python` (v3.11.14)
- **site-packages**: `/usr/local/python3.11.14/lib/python3.11/site-packages`（需加入 PYTHONPATH 才能 import modelscope 等包）
- **vllm 上游**: `/data/vllm` (v0.20.2, clean)
- **vllm-ascend 工作目录**: `/data/vllm-ascend` (0.19.1rc2, 有未提交改动)
- **模型**: `/data/models/MiMo-7B-Base/` (14.9GB, 4个分片已全部下载)

## 网络 / 代理

- 不再使用全局代理 (`/etc/profile.d/proxy.sh` → 已禁用)
- 按需代理: `source /etc/profile.d/proxy-helper.sh` → `proxy <cmd>` 或 `proxy-on`/`proxy-off`
- SSH 反向隧道: `ssh -R 7897:127.0.0.1:7897 server` → Windows Clash 代理

## 下载经验

- **ModelScope 最快**: 阿里云国内，直连 10-12 MB/s，不需要代理
- **hf-mirror.com 慢**: 索引在国内但文件重定向到 AWS S3 美国 (0.2-0.5 MB/s)
- ModelScope SDK: `snapshot_download("XiaomiMiMo/MiMo-7B-Base", cache_dir="/data/models")`
- Python 路径修复: `export PYTHONPATH="/usr/local/python3.11.14/lib/python3.11/site-packages:$PYTHONPATH"`

## 任务状态

### 任务1: --speculative-config → mimo_mtp/ernie_mtp (✅ 已完成，3 commits)
- 详见 [SKILL.md](SKILL.md)
- 3 commits on `feat/speculative-config-mtp`:
  1. `2166e4511` feat: mimo_mtp/ernie_mtp core adaptation
  2. `920705c88` fix: MTP + async_scheduling on Ascend NPU
  3. `214515e98` fix: remove defensive sync points
- 变更文件 (9 files, +341/-50):
  - `vllm_ascend/patch/worker/patch_mimo_mtp.py` (新增)
  - `vllm_ascend/patch/worker/patch_ernie_mtp.py` (新增)
  - `vllm_ascend/patch/worker/__init__.py` (修改)
  - `vllm_ascend/spec_decode/__init__.py` (修改)
  - `vllm_ascend/spec_decode/eagle_proposer.py` (修改)
  - `vllm_ascend/spec_decode/utils.py` (修改)
  - `vllm_ascend/worker/model_runner_v1.py` (修改)
- 性能: MTP n=1 +35%~+59% vs non-MTP

### 任务2: --attention-config → use_prefill_query_quantization (🔴 未开始)
- 难度: 高

### 其他待处理
- Triton-Ascend 驱动激活问题 (HAS_TRITON=False) — 阻塞端到端推理验证
- 模型已下载完成，可随时进行 import 验证和 MTP 推理测试

## 常用命令

```bash
# 验证 import
cd /data/vllm-ascend && /usr/local/python3.11.14/bin/python -c "
from vllm_ascend.patch.worker.patch_mimo_mtp import AscendMiMoMultiTokenPredictorLayer
print('mimo_mtp OK')
"

# 预提交检查
cd /data/vllm-ascend && git diff --stat

# 查看下载速度
ls -lh /data/models/MiMo-7B-Base/*.safetensors
```
