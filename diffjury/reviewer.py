import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .diff_parser import ParsedDiff
from .finding_identity import canonical_cwe, canonical_identity
from .models import Finding, Severity


class Reviewer(ABC):
    name = "reviewer"

    @abstractmethod
    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        raise NotImplementedError


class LocalRuleReviewer(Reviewer):
    name = "local-rules"
    domains = ("security", "reliability", "correctness")

    RULES = [
        (
            "SEC-EVAL",
            Severity.CRITICAL,
            re.compile(r"\b(eval|exec)\s*\("),
            "动态代码执行可能导致注入",
            "新增代码调用了动态执行函数；当参数可被外部影响时，攻击者可能执行任意代码。",
            "移除动态执行；使用显式解析器、命令映射表或严格白名单处理输入。",
            "加入恶意表达式与边界输入测试，断言输入不会被当作代码执行。",
        ),
        (
            "SEC-SUBPROCESS-SHELL",
            Severity.HIGH,
            re.compile(r"\bshell\s*=\s*True\b"),
            "Shell 调用存在命令注入风险",
            "shell=True 会扩大参数拼接造成命令注入的风险。",
            "使用参数数组并保持 shell=False；对允许值进行白名单验证。",
            "加入包含空格、分号与命令替换字符的输入测试。",
        ),
        (
            "SEC-HARDCODED-SECRET",
            Severity.HIGH,
            re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret|token)\b\s*=\s*['\"][^'\"]{4,}['\"]"),
            "疑似硬编码凭据",
            "凭据进入代码仓库后可能通过历史记录、构建日志或制品泄露。",
            "从密钥管理服务或环境变量读取，并立即轮换已经提交的凭据。",
            "测试缺少配置时安全失败，且日志不会输出凭据。",
        ),
        (
            "SEC-SQL-CONCAT",
            Severity.HIGH,
            re.compile(r"(?i)(execute|query)\s*\(\s*(f['\"]|['\"].*(\+|%))"),
            "SQL 语句疑似动态拼接",
            "将外部数据拼接到 SQL 中可能产生 SQL 注入。",
            "改用驱动提供的参数化查询与占位符。",
            "加入引号、注释符和布尔表达式等注入载荷测试。",
        ),
        (
            "REL-EMPTY-EXCEPT",
            Severity.MEDIUM,
            re.compile(r"^\s*except\s*(Exception\s*)?:\s*(pass)?\s*$"),
            "异常被宽泛捕获",
            "宽泛捕获会隐藏真实故障，使调用方误以为操作成功。",
            "仅捕获可处理的异常，记录必要上下文，并让不可恢复错误向上传播。",
            "加入依赖失败测试，断言错误可观察且不会返回伪成功。",
        ),
        (
            "REL-DEBUG-PRINT",
            Severity.LOW,
            re.compile(r"\b(print\s*\(|console\.log\s*\()"),
            "新增调试输出",
            "直接输出可能污染服务日志或意外暴露运行数据。",
            "删除调试输出，或改用带级别和脱敏策略的结构化日志。",
            "验证正常请求不会产生包含敏感值的非预期输出。",
        ),
    ]

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        findings: List[Finding] = []
        seen = set()
        for line in parsed.added_lines:
            if line.path.endswith((".lock", ".min.js", ".map")):
                continue
            for rule_id, severity, pattern, title, explanation, fix, test in self.RULES:
                if pattern.search(line.content) and (rule_id, line.path, line.line) not in seen:
                    seen.add((rule_id, line.path, line.line))
                    findings.append(
                        Finding(
                            rule_id=rule_id,
                            cwe=canonical_cwe(rule_id),
                            severity=severity,
                            title=title,
                            explanation=explanation,
                            path=line.path,
                            line=line.line,
                            evidence=line.content.strip()[:240],
                            fix=fix,
                            test=test,
                            confidence=0.9,
                            evidence_refs=[{
                                "evidence_id": "local-rule:%s" % hashlib.sha256(
                                    (rule_id + line.path + str(line.line) + line.content).encode("utf-8")
                                ).hexdigest()[:16],
                                "tool": "local-rule-scanner",
                                "rule_id": rule_id,
                                "path": line.path,
                                "line": line.line,
                            }],
                            source="local-rule-scanner",
                        )
                    )
        return findings


class DomainRuleReviewer(Reviewer):
    """Independent deterministic specialist backed by an explicit rule policy."""

    rule_ids = frozenset()
    domains = ()

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        findings: List[Finding] = []
        seen = set()
        rules = [item for item in LocalRuleReviewer.RULES if item[0] in self.rule_ids]
        for line in parsed.added_lines:
            if line.path.endswith((".lock", ".min.js", ".map")):
                continue
            for rule_id, severity, pattern, title, explanation, fix, test in rules:
                identity = (rule_id, line.path, line.line)
                if pattern.search(line.content) and identity not in seen:
                    seen.add(identity)
                    findings.append(Finding(
                        rule_id=rule_id, cwe=canonical_cwe(rule_id), severity=severity, title=title,
                        explanation=explanation, path=line.path, line=line.line,
                        evidence=line.content.strip()[:240], fix=fix, test=test,
                        confidence=0.9,
                        evidence_refs=[{
                            "evidence_id": "local-rule:%s" % hashlib.sha256(
                                (rule_id + line.path + str(line.line) + line.content).encode("utf-8")
                            ).hexdigest()[:16],
                            "tool": "local-rule-scanner", "rule_id": rule_id,
                            "path": line.path, "line": line.line,
                        }],
                        source="local-rule-scanner",
                    ))
        return findings

class SecurityRuleReviewer(DomainRuleReviewer):
    name = "security-agent"
    domains = ("security", "authorization")
    rule_ids = frozenset({
        "SEC-EVAL", "SEC-SUBPROCESS-SHELL", "SEC-HARDCODED-SECRET",
        "SEC-SQL-CONCAT",
    })


class ReliabilityRuleReviewer(DomainRuleReviewer):
    name = "reliability-agent"
    domains = ("reliability", "correctness", "regression")
    rule_ids = frozenset({"REL-EMPTY-EXCEPT", "REL-DEBUG-PRINT"})


class CompositeReviewer(Reviewer):
    name = "composite"

    def __init__(self, reviewers: List[Reviewer]):
        self.reviewers = reviewers
        self.name = "+".join(item.name for item in reviewers)

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        merged: Dict[Any, Finding] = {}
        errors = []
        for reviewer in self.reviewers:
            try:
                for finding in reviewer.review(diff, parsed):
                    key = (
                        finding.path, finding.line,
                        canonical_identity(finding.rule_id, finding.cwe),
                    )
                    merged[key] = finding
            except Exception as exc:
                errors.append(exc)
        if not merged and errors and len(errors) == len(self.reviewers):
            raise errors[0]
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        return sorted(merged.values(), key=lambda item: (order[item.severity], item.path, item.line))
