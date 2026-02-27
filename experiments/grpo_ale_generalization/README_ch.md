# GRPO ALE-Bench 泛化实验

## 这个实验要回答什么问题？

**核心问题**：在一组开放式算法题上做 RL（GRPO）训练后，模型是否能泛化到未见过的新题目？

**实验设计**：
- 从 ALE-Bench 中选 12 个问题，划分为 8 个训练题 / 4 个测试题
- 训练阶段：在训练题上进行带迭代改进（iterative refinement）的 GRPO 训练
- 评测阶段：在测试题上比较 base 模型与 finetuned 模型
- 指标：测试集上“最佳分数均值差”（finetuned - base）

## 系统架构

```
训练容器（8x H800）                  开发机（10.214.54.87）
+---------------------------+         +----------------------+
| 模型 + LoRA               |  HTTP   | FastAPI Evaluator    |
| GRPO Trainer              | ------> | Docker（编译/运行）  |
| Reward Function           |  /eval  | Problem Cache        |
| Dataset                   |         |                      |
+---------------------------+         +----------------------+
```

- **训练容器**：运行模型、生成代码、计算 GRPO 更新
- **开发机**：通过 Docker 进行编译、执行与判题
- 两端通过 HTTP 通信（训练端不做 Docker-in-Docker）

## 快速开始

### 1）部署 evaluator（在开发机 10.214.54.87 上）

```bash
bash experiments/grpo_ale_generalization/remote_evaluator/setup_devhost.sh
bash experiments/grpo_ale_generalization/remote_evaluator/start_devhost.sh
```

### 2）选择问题（在训练容器上）

```bash
# 查看可用问题
python experiments/grpo_ale_generalization/tools/list_problems.py

# 编辑 exp.yaml，填入 12 个 problem ID
vi experiments/grpo_ale_generalization/configs/exp.yaml
```

### 3）冒烟测试

```bash
# 快速测试（只做本地检查，不依赖 evaluator）
python experiments/grpo_ale_generalization/smoke_test.py --quick

# 完整测试（需要 evaluator 已启动）
python experiments/grpo_ale_generalization/smoke_test.py
```

### 4）生成训练数据

```bash
python experiments/grpo_ale_generalization/data/make_dataset.py \
  --config experiments/grpo_ale_generalization/configs/exp.yaml
```

### 5）开始训练

```bash
# 冒烟训练（4 step，最小配置）
bash experiments/grpo_ale_generalization/train.sh --smoke

# 正式训练
bash experiments/grpo_ale_generalization/train.sh
```

### 6）运行评估

```bash
# 只评估 base 模型
python experiments/grpo_ale_generalization/eval.py \
  --config experiments/grpo_ale_generalization/configs/exp.yaml \
  --base_only

# 评估 base + finetuned
python experiments/grpo_ale_generalization/eval.py \
  --config experiments/grpo_ale_generalization/configs/exp.yaml \
  --ckpt experiments/grpo_ale_generalization/checkpoints/step_200
```

## 配置说明

实验主配置集中在一个文件：`configs/exp.yaml`

切换问题时，通常只需修改：

```yaml
problem_pool:
  - ahc001
  - ahc002
  # ... 共 12 个
```

划分方式：
- 自动划分：按 seed=42 打乱后，前 8 个 train，后 4 个 test
- 手动划分：直接指定 `train_problem_ids` 与 `test_problem_ids`

```yaml
train_problem_ids: [ahc001, ahc002, ...]
test_problem_ids: [ahc009, ahc010, ...]
```

## 奖励设计（Reward）

### 公式

对迭代改进第 t 轮：

```
若失败（NO_CODE, COMPILE_ERROR, RE, TLE, MLE, WA）：
    reward_t = penalty[fail_type]

若成功（OK）：
    raw_score = K 个 seed 上的绝对分均值
    adjusted  = -raw_score（MINIMIZE）或 raw_score（MAXIMIZE）
    norm      = zscore_ema_clip(adjusted)         # 按问题做 z-score
    reward_t  = clip(norm_after - norm_before)    # 增量奖励，范围 [-3, 3]
```

### 失败惩罚（默认）

| 失败类型 | 默认惩罚 |
|---------|---------|
| NO_CODE | -2.0 |
| COMPILE_ERROR | -1.5 |
| RUNTIME_ERROR (RE) | -1.0 |
| TIME_LIMIT_EXCEEDED (TLE) | -1.0 |
| MEMORY_LIMIT_EXCEEDED (MLE) | -1.0 |
| WRONG_ANSWER (WA) | -0.5 |

### Z-Score 归一化（按 problem）

```
ema_mean = (1 - alpha) * ema_mean + alpha * adjusted_score
ema_var  = (1 - alpha) * ema_var  + alpha * (adjusted_score - ema_mean)^2
zscore   = (adjusted_score - ema_mean) / max(sqrt(ema_var), 1e-6)
clipped  = clip(zscore, [-3, 3])
```

**设计动机**：
- 不同题分数尺度不同，EMA 有助于自适应
- 方向统一（MINIMIZE 取负）后，可统一为“越大越好”
- clipping 提升 RL 稳定性
- 增量奖励鼓励“逐轮改进”，而非只追求一次性输出

### 诊断日志

奖励函数每步会输出按题聚合统计，例如：

```
[Step 10] problem=ahc001 batch_size=4 mean_reward=0.523 mean_raw_score=12345.6 fail_rate=0.25 ema_mean=12000.0 ema_std=500.00 cumul_fail_rate=0.15
```

## Iterative Refinement（迭代改进）

每个 episode 有 R 轮（默认 3）：

1. **Round 1**：根据题面生成代码 -> 评测 -> 获取反馈
2. **Round 2..R**：将反馈 + 当前最佳代码作为上下文继续改进 -> 评测 -> 更新 best

轨迹会维护：
- `best_code`：当前最优代码
- `best_score_raw`：当前最优代码的原始分数
- `last_feedback`：上一轮失败类型、分数、时间、内存、报错摘要

每一轮形成一个 GRPO 样本：
`(prompt=当前状态, response=新代码, reward=分数增量)`

## 训练配置建议

### 冒烟配置（快速验证）
- 4 steps，batch=2，group=2，R=2，K=1
- 大约 10 分钟可验证全链路是否可跑通

### 正式配置
- 200 steps，batch=8，group=4，R=3，K=3
- LoRA rank=16，bf16，gradient checkpointing
- 推荐 8x H800 + FSDP

### 性能瓶颈：Remote Evaluator

通常瓶颈在开发机 Docker 执行，可尝试：
- 提高开发机 `ALE_EVAL_MAX_CONCURRENT`（默认 4）
- 提高 `exp.yaml` 中 `client_max_concurrent`（默认 4）
- 使用 `lite_version: true` 加速评测
- 降低 `train_k_seeds`（每次评测 seed 更少）
- 开训前先 warmup 全部问题（`make_dataset.py` 会做）

## 目录结构

```
experiments/grpo_ale_generalization/
├── configs/
│   ├── exp.yaml              # 实验主配置（优先改这个）
│   └── verl_grpo.yaml        # VeRL 训练配置
├── data/
│   ├── make_dataset.py       # 数据集生成脚本
│   └── train_data/           # 生成的数据（已 gitignore）
├── reward/
│   ├── ale_reward.py         # VeRL 自定义奖励函数
│   ├── code_extract.py       # 从模型输出中抽取代码
│   └── prompt_builder.py     # 迭代改进提示词构造
├── remote_evaluator/
│   ├── evaluator_server.py   # FastAPI evaluator 服务
│   ├── setup_devhost.sh      # 开发机初始化脚本
│   ├── start_devhost.sh      # 开发机启动脚本
│   └── README_devhost.md     # 开发机部署说明
├── tools/
│   └── list_problems.py      # 列出可用 problem IDs
├── train.sh                  # 训练入口
├── train_standalone.py       # 无 VeRL 依赖时的训练脚本
├── eval.py                   # 评估脚本
├── smoke_test.py             # 全流程冒烟测试
└── README.md                 # 本文件（中文版）
```

## 常见问题

### 1）Remote evaluator 无法访问
- 确认开发机 `10.214.54.87:8000` 已启动服务
- 检查网络和防火墙
- 可用 SSH 隧道：

```bash
ssh -L 8000:localhost:8000 user@10.214.54.87 -N &
```

### 2）开发机缺少 Docker 镜像

```bash
bash scripts/docker_pull_all.sh yimjk/ale-bench
docker pull rust:1.79.0-buster
```

### 3）`problem_pool` 为空

```bash
python experiments/grpo_ale_generalization/tools/list_problems.py
# 然后编辑 configs/exp.yaml，填入 12 个问题
```

### 4）训练很慢
- 常见瓶颈是远端评测，而非 GPU
- 查看 evaluator 日志：

```bash
tail -f experiments/grpo_ale_generalization/remote_evaluator/evaluator.log
```

- 提高并发、降低 seed 数、使用 lite_version

### 5）显存不足（OOM）
- 减小 `batch_size` 或 `group_size`
- 确保 `gradient_checkpointing` 开启
- 降低 `max_new_tokens`
- 使用更小 LoRA rank

## 验收标准

满足以下条件可认为实验链路正确：
- [x] `make_dataset` -> `train`（少量 step）-> `eval`（base+ft）端到端可跑通
- [x] `eval` 输出包含测试题上的 base/ft 对比与总体 delta
- [x] 仅修改 `exp.yaml` 即可完成 12 题切换与 8/4 划分
- [x] 训练期评测全部走 remote evaluator（训练容器不直接跑 Docker）
- [x] 训练日志能看到奖励归一化与失败惩罚诊断信息
