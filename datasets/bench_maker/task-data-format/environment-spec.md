# environment/、solution/、tests/ 目录规范

## data/ — 模拟数据

为 Agent 提供运行时可访问的模拟数据，使任务具备可复现性。

| 子目录 | 用途 | 文件格式 |
|--------|------|----------|
| `local_files/` | 模拟用户本地文件 | 任意格式（如 `report.pdf`, `sales_data.csv`, `main.py`） |
| `mock_web_pages/` | 离线网页快照 | `.html`（特定任务需要） |

**命名规范**：文件名使用小写字母、数字、下划线和连字符，不含空格。

## skills/ — CoPaw Skills

任务可用的 Skill 目录，每个 Skill 一个子目录：

```
skills/
├── himalaya/
│   └── SKILL.md
├── news/
│   └── SKILL.md
└── calendar-ops/
    ├── SKILL.md
    └── scripts/
        └── parse_ics.py
```

Skill 格式遵循 CoPaw Skill 标准（SKILL.md + 可选辅助文件）。

`copaw.required_skills` 和 `copaw.distractor_skills` 中声明的 Skill 必须在此目录中提供。

## config/ — CoPaw 配置

| 文件 | 必须 | 说明 |
|------|------|------|
| `SOUL.md` | 是 | Agent 人设，定义核心准则、边界、风格 |
| `AGENTS.md` | 否 | Agent 行为定义，多 Agent 场景时使用 |
| `config.json` | 否 | CoPaw 运行配置（模型选择、工具白名单等） |

`SOUL.md` 格式参考：

```markdown
---
summary: "SOUL.md 工作区模板"
read_when:
  - 手动引导工作区
---

_你的 Agent 标语_

## 核心准则
...
## 边界
...
## 风格
...
## 连续性
...
```

## Dockerfile

可选。当任务需要特殊运行环境时提供。不提供时使用默认基础镜像。

```dockerfile
FROM copaw-base:latest

RUN pip install pandas openpyxl
COPY data/ /app/data/
```

---

## solution/ 目录

提供任务的 Oracle 参考解，用于：
- 生成参考轨迹作为训练数据
- 验证测试用例的正确性
- 对比 Agent 输出与最优解的差距

```bash
#!/bin/bash
# solve.sh — Oracle 解法

himalaya list --folder INBOX --max 10 | jq '.[] | select(.flags | contains("unread"))'
# ... 更多步骤
```

**要求**：
- `solve.sh` 必须可独立执行（`chmod +x`）
- 脚本内注释关键步骤的意图
- 退出码 0 表示成功

---

## tests/ 目录

### test.sh

测试入口脚本，评测框架调用此脚本执行验证。

```bash
#!/bin/bash
set -e

python -m pytest test_outputs.py -v --tb=short
# 退出码: 0=通过, 非0=失败
```

### test_outputs.py

pytest 验证用例，检查 Agent 的实际产出。

```python
import json
import pytest
from pathlib import Path

def test_output_file_exists():
    assert Path("output/result.json").exists()

def test_output_correctness():
    with open("output/result.json") as f:
        result = json.load(f)
    assert result["status"] == "success"
    assert len(result["items"]) >= 3
```
