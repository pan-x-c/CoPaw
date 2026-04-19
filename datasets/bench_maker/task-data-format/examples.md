# 完整示例与数据集组织

## 示例 1：邮件分类任务

```
email_triage_0001_zh/
├── task.yaml
├── environment/
│   ├── data/
│   │   ├── mock_emails/
│   │   │   ├── 001_meeting_invite.eml
│   │   │   ├── 002_spam_ad.eml
│   │   │   ├── 003_urgent_client.eml
│   │   │   └── 004_newsletter.eml
│   ├── skills/
│   │   └── himalaya/
│   │       └── SKILL.md
│   └── config/
│       └── SOUL.md
├── solution/
│   └── solve.sh
└── tests/
    ├── test.sh
    └── test_outputs.py
```

对应 `task.yaml`：

```yaml
version: "2.0"

messages:
  - role: user
    content:
      - type: text
        text: "帮我分一下邮箱里的邮件，哪些需要回复、哪些是通知、哪些是垃圾邮件。把分类结果保存到 classification.json。"

metadata:
  task_id: "email_triage_0001_zh"
  author: "yunpeng"
  labels:
    capabilities:
      - "Tool_Calling"
      - "Email_Communication"
      - "Knowledge_Retrieval"
    scenario: "Office_Productivity/Email/Inbox_Triage"
    difficulty: "medium"
    step_type: "ReAct"
    hop_count: "L2"
    persona: "Default_Assistant"
    style_tone: "Professional"
    sysprompt_dependency: "None"

copaw:
  required_tools:
    - "himalaya"
  required_skills:
    - "himalaya"
  distractor_skills:
    - "calendar-ops"

environment:
  paths:
    - "environment/data/mock_emails/"
    - "environment/config/SOUL.md"

evaluation:
  eval_type: "hybrid"
  test_script: "tests/test.sh"
  dimensions:
    session_check: false
    llm_judge: true
  inputs:
    answer: ""
  llm_judge:
    grader_type:
      - "CorrectnessGrader"
    model: "qwen3-32b"
    threshold: 4
    language: "zh"
  env:
    required:
      - "DASHSCOPE_API_KEY"

setup: []
```

## 示例 2：多模态识别任务（无环境依赖）

```
mat_0001_en/
├── task.yaml
└── environment/
    └── config/
        └── SOUL.md
```

对应 `task.yaml`：

```yaml
version: "2.0"

messages:
  - role: user
    content:
      - type: text
        text: "From the attached image, identify the species of the plant shown. Provide the common and scientific name."
      - type: image
        image_url: /app/working/workspaces/default/local_files/plant_image.jpg

metadata:
  task_id: "mat_0001_en"
  author: "sunyuchang"
  labels:
    capabilities:
      - "Multimodal_Perception"
    scenario: "Education_Knowledge/General_QA"
    difficulty: "medium"
    step_type: "Zero_Shot"
    hop_count: "L1"

copaw:
  required_tools: []
  required_skills: []
  distractor_skills: []

environment:
  paths: []

evaluation:
  eval_type: ""
  test_script: ""
  dimensions:
    session_check: false
    llm_judge: true
  inputs:
    answer: "{'common_name': 'oak', 'scientific_name': 'Quercus'}"
  llm_judge:
    grader_type:
      - "CorrectnessGrader"
    model: "qwen3-32b"
    threshold: 4
    language: "en"
  env:
    required:
      - "DASHSCOPE_API_KEY"

setup: []
```

---

## 数据集级别组织

多个任务组成一个数据集：

```
dataset_name/
├── README.md
├── manifest.yaml
├── email_triage_0001_zh/
│   ├── task.yaml
│   └── environment/
├── code_debug_0001_en/
│   ├── task.yaml
│   └── environment/
└── ...
```

### manifest.yaml

```yaml
name: "copaw-eval-bench-v1"
version: "1.0"
created: "2026-04-14"
description: "CoPaw Agent 综合评测集"
total_tasks: 150
statistics:
  by_difficulty:
    easy: 30
    medium: 60
    hard: 45
    expert: 15
  by_scenario:
    Office_Productivity: 40
    Software_Engineering: 25
    Information_Retrieval: 20
    Safety_Compliance: 15
    Multimodal: 20
    Other: 30
tasks:
  - task_id: "email_triage_0001_zh"
  - task_id: "code_debug_0001_en"
  # ...
```

---

## 校验清单

制作或审核一个 task 时，逐项检查：

### task.yaml
- [ ] `version` 为 `"2.0"`
- [ ] `messages` 至少包含一条 role=user 的消息
- [ ] `task_id` 格式正确且全局唯一
- [ ] `labels` 已填写，且 `difficulty` 取值在 easy/medium/hard/expert 之中
- [ ] `copaw` 三个字段均已声明（即使为空列表）
- [ ] `copaw.required_skills` 和 `copaw.distractor_skills` 中的 Skill 均存在于 `environment/skills/`
- [ ] `evaluation.dimensions` 至少开启一种评测方式
- [ ] 用户 query 具备可完成性（不依赖不存在的数据/资源）

### environment/
- [ ] `data/` 中引用的文件均存在
- [ ] `skills/` 中的每个 Skill 含有效 `SKILL.md`
- [ ] `environment.paths` 中列出的路径均实际存在

### evaluation
- [ ] LLM judge 启用时，`llm_judge` 配置完整
- [ ] 脚本验证启用时，`test_script` 路径有效
- [ ] `inputs.answer` 或 `inputs.reference_file` 至少提供一个

### 命名与编码
- [ ] 所有文件 UTF-8 编码
- [ ] 文件名不含空格和特殊字符
- [ ] YAML 缩进统一为 2 空格
