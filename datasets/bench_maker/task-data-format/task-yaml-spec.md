# task.yaml 完整字段规范

## version

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `version` | string | 是 | 格式版本号，当前为 `"2.0"` |

## messages

用户输入的消息序列，模拟用户与 Agent 的对话起点。

```yaml
messages:
  - role: user
    content:
      - type: text
        text: "用户的文字消息"
      - type: image
        image_url: "图片路径（相对于运行环境根目录）"
```

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `messages` | list | 是 | 消息列表，至少包含一条 |
| `messages[].role` | string | 是 | 消息角色：`user` / `system` |
| `messages[].content` | list | 是 | 消息内容块列表，支持多模态 |
| `content[].type` | string | 是 | 内容类型：`text` / `image` |
| `content[].text` | string | 条件 | type=text 时必须 |
| `content[].image_url` | string | 条件 | type=image 时必须，图片路径 |

**多轮对话**示例：

```yaml
messages:
  - role: user
    content:
      - type: text
        text: "帮我搜索最新的AI论文"
  - role: user
    content:
      - type: text
        text: "重点关注多模态方向的"
```

## metadata

任务元信息，用于索引、筛选和统计。

```yaml
metadata:
  task_id: "email_triage_0001_zh"
  author: "zhangsan"
  labels:
    capabilities:
      - "Tool_Calling"
      - "Email_Communication"
    scenario: "Office_Productivity/Email/Inbox_Triage"
    difficulty: "hard"
    step_type: "ReAct"
    hop_count: "L2"
    persona: "Default_Assistant"
    style_tone: "Professional"
    sysprompt_dependency: "None"
```

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | 全局唯一任务 ID，格式：`{category_abbr}_{sequence}_{lang}` |
| `author` | string | 是 | 数据作者 |
| `labels` | object | 是 | 多维度结构化标签，详见标签体系文档 |

### task_id 命名规范

```
{category_abbr}_{sequence}_{lang}
```

| 部分 | 说明 | 示例 |
|------|------|------|
| `category_abbr` | 类别缩写，2-6 个小写字母 | `email`, `code`, `fin`, `safety`, `mat` |
| `sequence` | 4 位数字序号，从 0001 开始 | `0001`, `0042` |
| `lang` | ISO 639-1 语言代码 | `zh`, `en`, `ja` |

示例：`email_triage_0001_zh`, `code_debug_0023_en`, `mat_0001_en`

### difficulty 判定标准

| 等级 | 定义 | 典型特征 |
|------|------|----------|
| `easy` | 单步或 2 步即可完成 | 直接回答、单 API 调用 |
| `medium` | 3-5 步，需要中等推理 | 多工具协调、简单规划 |
| `hard` | >5 步，需要深度推理或多系统协作 | 复杂工作流、错误恢复 |
| `expert` | 需要领域专业知识 + 复杂执行 | 跨域分析、创造性综合 |

## copaw

CoPaw 框架运行时的特定配置。

```yaml
copaw:
  required_tools:
    - "himalaya"
    - "calendar_cli"
  required_skills:
    - "email-management"
    - "calendar-ops"
  distractor_skills:
    - "stock-analysis"
    - "video-editing"
```

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `required_tools` | list[string] | 是 | 完成任务必须的工具列表，空列表 `[]` 表示不依赖工具 |
| `required_skills` | list[string] | 是 | 完成任务必须的 Skill 列表。列出的 Skill 必须在 `environment/skills/` 目录中提供 |
| `distractor_skills` | list[string] | 是 | 干扰项 Skill，测试 Agent 的工具选择能力。列出的 Skill 同样必须在 `environment/skills/` 中提供 |

## environment

指定环境目录中需要在运行时挂载的文件路径。

```yaml
environment:
  paths:
    - "environment/data/local_files/report.pdf"
    - "environment/config/SOUL.md"
```

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `paths` | list[string] | 是 | 需要挂载到运行环境的文件/目录列表，路径相对于任务根目录。空列表 `[]` 表示无额外文件。 |

## evaluation

评测配置，定义如何判定任务完成情况。

```yaml
evaluation:
  eval_type: "script"
  test_script: "tests/test.sh"
  dimensions:
    session_check: false
    llm_judge: true
  inputs:
    answer: "{'key': 'expected_value'}"
  llm_judge:
    grader_type:
      - "CorrectnessGrader"
    model: "qwen3-32b"
    threshold: 4
    language: "zh"
  env:
    required:
      - "DASHSCOPE_API_KEY"
```

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `eval_type` | string | 是 | 评测方式：`script` / `llm_judge` / `hybrid` / `""` |
| `test_script` | string | 否 | 测试脚本路径，eval_type 含脚本验证时必须 |
| `dimensions.session_check` | bool | 是 | 是否检查完整会话轨迹 |
| `dimensions.llm_judge` | bool | 是 | 是否启用 LLM 判分 |
| `inputs` | object | 否 | 评测输入，key-value 形式 |
| `inputs.answer` | string | 否 | 标准答案（字符串形式，可为 JSON） |
| `inputs.reference_file` | string | 否 | 参考输出文件路径 |
| `llm_judge.grader_type` | list[string] | 条件 | LLM 判分类型，多选 |
| `llm_judge.model` | string | 条件 | 评分模型名称 |
| `llm_judge.threshold` | int | 条件 | 通过阈值，1-5 分制 |
| `llm_judge.language` | string | 条件 | 评分语言：`zh` / `en` |
| `env.required` | list[string] | 否 | 评测运行时需要的环境变量 |

### grader_type 详解

| 类型 | 评分维度 | 适用场景 |
|------|----------|----------|
| `CorrectnessGrader` | 答案是否正确、完整 | 有明确标准答案的任务 |
| `HallucinationGrader` | 回答是否存在幻觉 | 开放式生成任务 |
| `TrajectoryAccuracyGrader` | 目标是否完成，过程是否有冗余 | 轨迹质量检查 |
| `custom` | 自定义评分标准 | 需配合 `inputs` 中提供的 rubric |

## setup

任务运行前的初始化步骤。

```yaml
setup:
  - "setup.sh"
```

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `setup` | list[string] | 否 | 初始化脚本列表，按顺序执行。省略或空列表 `[]` 表示无需初始化。脚本路径相对于任务根目录。 |
