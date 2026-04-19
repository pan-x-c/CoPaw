"""
测试用例：skillhub_0141 — Skill 使用能力评估

场景：Agent 需要识别并使用正确的 skill 完成用户任务。
      required_skills: ['feishu-evolver-wrapper']
      distractor_skills: ['browser_cdp', 'cron']

评估维度：
  确定性检查（session_check）：
    skill_discovered   — Agent 是否发现并读取了所需 skill
    output_generated   — Agent 是否产出了实质性内容
  LLM 评估：
    trajectory_quality — 整体轨迹质量
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
    log_grader_score_line,
    read_session,
)
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderMode, GraderScore

WORKSPACE = os.environ.get(
    "COPAW_WORKSPACE_DIR", "/app/working/workspaces/default"
)

REQUIRED_SKILLS = ['feishu-evolver-wrapper']
SKILL_PATTERN = re.compile(r"(?i)(feishu[_\-]?evolver[_\-]?wrapper)")


def _get_session_texts():
    """提取 session 中的 assistant 文本和工具调用文本。"""
    session = read_session()
    if session is None:
        return "", ""
    all_assistant_text = extract_final_response(session)
    tool_calls = extract_tool_calls(session)
    all_tool_text = "\n".join(
        str(tc.get("input") or tc.get("arguments") or {})
        for tc in tool_calls
    )
    return all_assistant_text, all_tool_text


# ── 确定性检查函数 ─────────────────────────────────────────────────────


def _check_skill_discovered(response: str, **kwargs) -> GraderScore:
    """检查 Agent 是否发现并读取了所需的 skill。

    判断依据：
      - assistant 文本中提到了 skill 名称
      - 或工具调用中读取了 skill 相关文件（SKILL.md 等）
    """
    assistant_text, tool_text = _get_session_texts()
    combined = (assistant_text + "\n" + tool_text).lower()

    skill_mentioned = bool(SKILL_PATTERN.search(combined))

    skill_file_read = bool(
        re.search(r"skill\.md|skills?/", tool_text.lower())
    )

    if skill_mentioned and skill_file_read:
        return GraderScore(
            name="skill_discovered",
            score=1.0,
            reason="Agent 发现并读取了所需 skill",
        )
    if skill_mentioned:
        return GraderScore(
            name="skill_discovered",
            score=0.8,
            reason="Agent 提到了所需 skill，但未明确读取 SKILL.md",
        )
    if skill_file_read:
        return GraderScore(
            name="skill_discovered",
            score=0.5,
            reason="Agent 读取了 skill 文件，但未明确提及所需 skill 名称",
        )
    return GraderScore(
        name="skill_discovered",
        score=0.0,
        reason="Agent 未发现所需 skill",
    )


def _check_output_generated(response: str, **kwargs) -> GraderScore:
    """检查 Agent 是否产出了实质性内容（文本回复或文件输出）。"""
    assistant_text, tool_text = _get_session_texts()

    has_substantial_text = len(assistant_text.strip()) > 200

    file_write_patterns = [
        r"write_file",
        r"send_file",
        r"file_path.*\.(?:md|csv|xlsx|html|json|pdf|txt|py)",
    ]
    has_file_output = any(
        re.search(p, tool_text.lower()) for p in file_write_patterns
    )

    if has_substantial_text and has_file_output:
        return GraderScore(
            name="output_generated",
            score=1.0,
            reason="Agent 产出了实质性文本回复和文件输出",
        )
    if has_file_output:
        return GraderScore(
            name="output_generated",
            score=0.8,
            reason="Agent 生成了输出文件",
        )
    if has_substantial_text:
        return GraderScore(
            name="output_generated",
            score=0.6,
            reason=f"Agent 产出了文本回复 ({len(assistant_text)} chars)，但未生成文件",
        )
    return GraderScore(
        name="output_generated",
        score=0.0,
        reason="Agent 未产出实质性内容",
    )


# ── Test Classes ──────────────────────────────────────────────────────


class TestSkillDiscovered:
    """确定性检查：Agent 是否发现并使用了所需 skill"""

    @pytest.mark.asyncio
    async def test_skill_discovered(self):
        grader = FunctionGrader(
            func=_check_skill_discovered,
            name="skill_discovered",
            mode=GraderMode.POINTWISE,
        )
        result = await grader.aevaluate(response="")
        assert_grader_score(result, min_score=0.5, label="Skill 发现与读取")


class TestOutputGenerated:
    """确定性检查：Agent 是否产出了实质性内容"""

    @pytest.mark.asyncio
    async def test_output_generated(self):
        grader = FunctionGrader(
            func=_check_output_generated,
            name="output_generated",
            mode=GraderMode.POINTWISE,
        )
        result = await grader.aevaluate(response="")
        assert_grader_score(result, min_score=0.5, label="内容产出")


class TestTrajectoryQuality:
    """LLM 轨迹评估（计入得分）"""

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="未设置 DASHSCOPE_API_KEY，跳过轨迹评估",
    )
    @pytest.mark.asyncio
    async def test_trajectory_quality(self):
        session = read_session()
        assert session is not None, "Session 读取失败"
        result = await evaluate_trajectory(session)
        assert_grader_score(result, min_score=3.0, label="轨迹质量")


def _extract_user_query(session: dict) -> str:
    for turn in session.get("agent", {}).get("memory", {}).get("content", []):
        if not turn or len(turn) < 1:
            continue
        msg = turn[0]
        if msg.get("role") == "user":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
                if isinstance(block, str):
                    return block
    return ""


class TestHallucinationCheck:
    """LLM 幻觉评估：基于工具输出判断 Agent 回答是否存在捏造"""

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="未设置 DASHSCOPE_API_KEY，跳过幻觉评估",
    )
    @pytest.mark.asyncio
    async def test_hallucination(self):
        session = read_session()
        assert session is not None, "Session 读取失败"
        query = _extract_user_query(session)
        assert query, "无法从 session 提取用户 query"
        result = await evaluate_file_correctness(
            session, query, reference_response=""
        )
        assert_grader_score(result, min_score=3.0, label="幻觉检测")
