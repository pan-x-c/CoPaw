---
name: task-data-format
description: >-
  CoPaw Agent Task 数据的标准化格式规范。定义 task 目录结构、task.yaml 字段、
  environment 配置、评测方式等。当用户提到 task 数据格式、task.yaml、
  Agent 任务制作、评测数据、Benchmark 数据、任务目录结构、数据标准化 时使用此 skill。
---

# CoPaw Agent Task 数据标准化格式 v1.0

一个 task 目录 = 一个完整的、可独立运行的 Agent 任务。

## 目录结构总览

```
{task_id}/
├── task.yaml                    # 必须 — 任务定义、元数据、评测配置
├── environment/                 # 必须 — 运行环境
│   ├── Dockerfile               #   容器环境配置（按需）
│   ├── setup.sh                 #   可选 - 环境启动命令
│   ├── data/                    #   模拟输入数据
│   │   ├── local_files/         #     用户本地文件（report.pdf, data.csv 等）
│   │   └── mock_web_pages/      #     离线网页快照
│   ├── skills/                  #   可用的 CoPaw Skills
│   │   └── {skill-name}/SKILL.md
│   └── config/                  #   CoPaw 配置
│       ├── SOUL.md              #     Agent 人设
│       ├── AGENTS.md            #     Agent 行为定义
│       └── config.json          #     CoPaw 运行配置
├── solution/                    # 可选 — 参考解
│   └── solve.sh
└── tests/                       # 可选 — 验证测试
    ├── test.sh
    └── test_outputs.py
```

| 组件 | 必须/可选 | 说明 |
|------|-----------|------|
| `task.yaml` | **必须** | 任务唯一入口 |
| `environment/` | **必须** | 至少包含空目录 |
| `environment/config/` | 按需 | 可为空；通常含 SOUL.md、AGENTS.md、config.json |
| `environment/data/` | 按需 | 不涉及外部数据时可省略 |
| `environment/skills/` | 按需 | 不使用 Skill 时可省略 |
| `solution/` | 可选 | Oracle 参考解 |
| `tests/` | 可选 | 评测框架自动执行 |

## task.yaml 顶层结构

```yaml
version: "2.0"
messages: [...]          # 用户消息序列
metadata:                # 任务元信息（task_id, author, labels）
  task_id: "email_triage_0001_zh"
  author: "zhangsan"
  labels: { ... }        # 多维度标签（详见标签体系 skill）
copaw:                   # CoPaw 框架配置
  required_tools: [...]
  required_skills: [...]       # 须在 environment/skills/ 中提供
  distractor_skills: [...]     # 须在 environment/skills/ 中提供
environment:             # 运行时挂载文件
  paths: [...]
evaluation:              # 评测配置
  eval_type: "script" | "llm_judge" | "hybrid" | ""
  test_script: "tests/test.sh"
  dimensions:
    session_check: false
    llm_judge: true
  inputs:
    answer: "..."
  llm_judge:
    grader_type: [...]   # 多选
    model: "qwen3-32b"
    threshold: 4
    language: "zh"
setup: [...]             # 初始化脚本列表
```

## 关键字段速查

### task_id 命名

```
{category_abbr}_{sequence}_{lang}
```

- `category_abbr`: 2-6 小写字母（`email`, `code`, `fin`, `safety`, `mat`）
- `sequence`: 4 位数字（`0001`）
- `lang`: ISO 639-1（`zh`, `en`, `ja`）

### messages 内容类型

| type | 必须字段 | 说明 |
|------|----------|------|
| `text` | `text` | 文字消息 |
| `image` | `image_url` | 图片路径（相对于运行环境根目录） |

### labels（多维度标签）

标签体系详见 [task-label-taxonomy skill](../task-label-taxonomy/SKILL.md)，主要字段：

| 字段 | 说明 |
|------|------|
| `capabilities` | 原子能力（多选） |
| `scenario` | 领域场景路径（单选） |
| `difficulty` | `easy` / `medium` / `hard` / `expert` |
| `step_type` | `Zero_Shot` / `CoT` / `ReAct` / `Plan_and_Execute` / `Multi_Agent` |
| `hop_count` | `L1`(1-2步) / `L2`(3-5步) / `L3`(>5步) |
| `persona` | 角色设定 |
| `style_tone` | 语气风格 |
| `sysprompt_dependency` | System Prompt 依赖类型 |

### grader_type（多选）

| 类型 | 评分维度 | 适用场景 |
|------|----------|----------|
| `CorrectnessGrader` | 答案是否正确、完整 | 有明确标准答案的任务 |
| `HallucinationGrader` | 回答是否存在幻觉 | 开放式生成任务 |
| `TrajectoryAccuracyGrader` | 目标是否完成，过程是否有冗余 | 轨迹质量检查 |
| `custom` | 自定义评分标准 | 需配合 `inputs` 中提供的 rubric |

### difficulty 判定

| 等级 | 定义 | 典型特征 |
|------|------|----------|
| `easy` | 单步或 2 步即可完成 | 直接回答、单 API 调用 |
| `medium` | 3-5 步，中等推理 | 多工具协调、简单规划 |
| `hard` | >5 步，深度推理或多系统协作 | 复杂工作流、错误恢复 |
| `expert` | 领域专业知识 + 复杂执行 | 跨域分析、创造性综合 |

## 详细参考

- task.yaml 完整字段规范：[task-yaml-spec.md](task-yaml-spec.md)
- environment/solution/tests 目录规范：[environment-spec.md](environment-spec.md)
- 完整示例、数据集组织与校验清单：[examples.md](examples.md)

## 操作指引

### 创建新任务

1. 以 `task_id` 为目录名创建任务目录
2. 编写 `task.yaml`（参考上方结构或 [完整字段规范](task-yaml-spec.md)）
3. 按需填充 `environment/`（data、skills、config）
4. 可选：添加 `solution/` 和 `tests/`
5. 按 [校验清单](examples.md#校验清单) 逐项检查

### 审核现有任务

1. 对照 [校验清单](examples.md#校验清单) 检查完整性
2. 确认 `labels` 符合标签体系规范
3. 确认 `copaw` 中声明的 skill 在 `environment/skills/` 中存在
4. 确认用户 query 具备可完成性
