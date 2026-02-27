# 环境搭建与 Smoke Test 复现指南

> 记录于 2026-02-27，基于 4x H800 训练机 + 远程评测机 (10.214.54.87) 的环境。

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 训练机 | Linux, conda 已安装, 可通过 `source ~/.bashrc` 激活 |
| 远程评测机 | 10.214.54.87，已部署 evaluator_server，端口 8003 |
| 网络 | 训练机访问外网需配置代理；访问评测机走内网直连 |

---

## Step 0: 验证远程评测机

在训练机上执行：

```bash
curl -s http://10.214.54.87:8003/health
```

预期返回：

```json
{"status":"ok","max_concurrent":"12"}
```

如果不通，需先在评测机上启动服务（参见 `remote_evaluator/README_devhost.md`）。

---

## Step 1: 配置代理

训练机访问外网（conda/pip 下载包）需要代理，但访问评测机(10.214.54.87)不能走代理：

```bash
export no_proxy=localhost,127.0.0.1,10.52.104.88,10.214.54.87
export http_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
export https_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
```

> **重要**：每次开新终端都需要重新设置，或者写入 `~/.bashrc`。

---

## Step 2: 创建 conda 环境

```bash
source ~/.bashrc
conda create -n ale_bench python=3.12 -y
```

耗时约 1-2 分钟。如果 conda 连接超时，检查 Step 1 的代理是否已设置。

---

## Step 3: 激活环境并安装依赖

```bash
source ~/.bashrc
conda activate ale_bench
```

进入项目根目录，以 editable 模式安装 ale_bench 及所有依赖：

```bash
cd /root/paddlejob/workspace/xjk_exp/ALE-Bench
pip install -e .
```

这会自动安装以下核心依赖：
- `httpx` (HTTP 客户端，与评测机通信)
- `pyyaml` (配置文件解析)
- `pydantic` (数据模型)
- `huggingface-hub` (题目数据下载)
- `docker` (SDK，训练机不直接用但包含在依赖中)
- `polars`, `pillow`, `cairosvg` 等

确认安装成功：

```bash
python3 -c "import ale_bench; print('ale_bench:', ale_bench.__version__)"
python3 -c "import yaml; import httpx; print('yaml + httpx: OK')"
```

预期输出：

```
ale_bench: 1.3.0.post5.dev0+7116b91
yaml + httpx: OK
```

---

## Step 4: 运行 Smoke Test

### 4a) Quick 模式（仅本地模块，不需要评测机）

```bash
cd /root/paddlejob/workspace/xjk_exp/ALE-Bench
python3 experiments/grpo_ale_generalization/smoke_test.py --quick
```

预期输出：

```
[TEST 5] Code extraction...
  PASS: Code extraction works correctly.

[TEST 6] Reward module...
  PASS: Penalty table: {'NO_CODE': -2.0, 'COMPILE_ERROR': -1.5, ...}

[TEST 7] Prompt builder...
  PASS: Prompt builder works correctly.

==================================================
SMOKE TEST: 3/3 passed
ALL TESTS PASSED
==================================================
```

### 4b) Full 模式（含远程评测机验证）

```bash
python3 experiments/grpo_ale_generalization/smoke_test.py
```

预期输出：

```
Smoke test problem: ahc001
Evaluator URL: http://10.214.54.87:8003

[TEST 5] Code extraction...           PASS
[TEST 6] Reward module...             PASS
[TEST 7] Prompt builder...            PASS
[TEST 1] Health check...              PASS
[TEST 2] Warmup problem 'ahc001'...   PASS (1.2s)
[TEST 3] Evaluate broken code...      PASS (COMPILE_ERROR)
[TEST 4] Evaluate simple valid code...PASS (WA, score=0.0)

==================================================
SMOKE TEST: 7/7 passed
ALL TESTS PASSED
==================================================
```

> TEST 4 返回 WA (Wrong Answer) 和 score=0.0 是正常的——测试用的是一段无意义代码，能正确编译运行并被判题器评分即代表链路通畅。

---

## 常见问题

### conda create 超时

原因：未配置代理。解决：执行 Step 1 后重试。

### pip install -e . 失败

检查是否在项目根目录 `/root/paddlejob/workspace/xjk_exp/ALE-Bench` 下执行，该目录应包含 `pyproject.toml`。

### Smoke test TEST 1 Health check FAIL

评测机未启动或网络不通。确认：
1. `curl -s http://10.214.54.87:8003/health` 是否返回 `{"status":"ok"}`
2. `no_proxy` 中是否包含 `10.214.54.87`（否则请求会走代理导致不通）

### Smoke test TEST 2 Warmup 超时

首次 warmup 需要从 HuggingFace 下载题目数据并编译 Rust 工具，可能需要数分钟。后续调用会命中缓存。

---

## 环境总结

| 组件 | 版本/路径 |
|------|----------|
| conda 环境名 | `ale_bench` |
| Python | 3.12 |
| ale_bench | 1.3.0.post5.dev0+7116b91 (editable) |
| 项目目录 | `/root/paddlejob/workspace/xjk_exp/ALE-Bench` |
| 评测机地址 | `http://10.214.54.87:8003` |
| 配置文件 | `experiments/grpo_ale_generalization/configs/exp.yaml` |

---

## 后续步骤（参考）

环境验证通过后，后续流程为：

1. **生成数据集**：`python3 experiments/grpo_ale_generalization/data/make_dataset.py --config experiments/grpo_ale_generalization/configs/exp.yaml`
2. **训练 smoke test**：`bash experiments/grpo_ale_generalization/train.sh --smoke`
3. **正式训练**：`bash experiments/grpo_ale_generalization/train.sh`
4. **评测**：`python3 experiments/grpo_ale_generalization/eval.py --config ... --ckpt ...`
