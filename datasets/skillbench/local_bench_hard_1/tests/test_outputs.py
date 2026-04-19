"""
测试用例：local_bench_hard_1 - Skill 使用能力评估

场景：Agent 需要正确评估任务难度，并使用 make_plan 技能制定合理的计划，最终产出实质性文本回复以及调研报告。
      required_skills: ['make_plan']

评估维度：
  确定性检查（session_check）:
    skill_discovered   - Agent 是否发现并读取了所需的 skill（make_plan）
    tool_usage         - Agent 是否正确使用了 list_agents 和 chat_with_agent 工具来协作完成任务
"""

import os
import re

import pytest

from copaw_eval import (
    assert_check,
    assert_grader_score,
    evaluate_file_correctness,
    evaluate_trajectory,
    extract_final_response,
    extract_tool_calls,
    read_session,
)
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderMode, GraderScore

WORKSPACE = os.environ.get(
    "COPAW_WORKSPACE_DIR", "/app/working/workspaces/default"
)

REQUIRED_SKILLS = ['make_plan']
SKILL_PATTERN = re.compile(r"(?i)(make[_\-]?plan)")

def _get_session_info():
    """提取 session 中的 assistant 文本和工具调用信息。"""
    session = read_session()
    if session is None:
        return "", ""
    all_assistant_text = extract_final_response(session)
    tool_calls = extract_tool_calls(session)
    return all_assistant_text, tool_calls

def _check_skill_and_tool_usage(response: str, **kwargs) -> GraderScore:
    """检查 Agent 是否发现并读取了所需的 skill，并正确调用工具。

    判断依据：
      - 正确读取 skill
      - 正确使用 list_agents 和 chat_with_agent 工具来协作完成任务
    """
    assistant_text, tool_calls = _get_session_info()
    combined = (assistant_text + "\n" + "\n".join(
        str(tc.get("input") or tc.get("arguments") or {}) for tc in tool_calls
    )).lower()

    skill_mentioned = bool(SKILL_PATTERN.search(combined))

    skill_file_read = bool(
        re.search(r"skill\.md|skills?/", combined)
    )

    if not skill_mentioned or not skill_file_read:
        return GraderScore(
            name="skill_discovered",
            score=0.0,
            reason="Agent 未能发现并读取所需 Skill",
        )

    list_agents_used = any(
        tc.get("name") == "list_agents" for tc in tool_calls
    )
    if not list_agents_used:
        return GraderScore(
            name="tool_usage",
            score=0.0,
            reason="Agent 未使用 list_agents 工具来发现可用 agents",
        )
    # 检查是否使用了 chat_with_agent 工具来与其他 agent 协作
    # 并确保参数中包含了其他 agent 的信息（如 agent_id、message 等）
    chat_with_agent_used = any(
        tc.get("name") == "chat_with_agent" for tc in tool_calls
    )
    if not chat_with_agent_used:
        return GraderScore(
            name="tool_usage",
            score=0.0,
            reason="Agent 使用了 list_agents 工具，但未使用 chat_with_agent 工具来协作",
        )
    reason = "Agent 正确使用了 list_agents 和 chat_with_agent 工具"
    chat_with_agent_valid = True
    # 检查 chat_with_agent 工具的参数是否包含其他 agent 的信息
    if chat_with_agent_used:
        for tc in tool_calls:
            if tc.get("name") == "chat_with_agent":
                arguments = tc.get("arguments") or {}
                to_agent = arguments.get("to_agent")
                text = arguments.get("text")
                if to_agent != "default":
                    chat_with_agent_valid = False
                    reason = "Agent 使用了 chat_with_agent 工具，但未正确指定目标 agent"
                    break
                if text is None or not isinstance(text, str) or len(text.strip()) == 0:
                    chat_with_agent_valid = False
                    reason = "Agent 使用了 chat_with_agent 工具，但未提供有效的消息内容"
                    break

    if chat_with_agent_valid:
        return GraderScore(
            name="tool_usage",
            score=1.0,
            reason=reason,
        )
    else:
        return GraderScore(
            name="tool_usage",
            score=0.2,
            reason=reason,
        )


class TestSkillAndToolUsage:
    """确定性检查：Agent 是否发现并使用了所需 skill，并正确调用工具"""

    @pytest.mark.asyncio
    async def test_skill_discovered(self):
        grader = FunctionGrader(
            func=_check_skill_and_tool_usage,
            name="skill_and_tool_usage",
            mode=GraderMode.POINTWISE,
        )
        result = await grader.aevaluate(response="")
        assert_grader_score(result, min_score=0.5, label="Skill 与工具使用")
