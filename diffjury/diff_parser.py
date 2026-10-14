import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import ChangedLine


@dataclass
class ParsedDiff:
    files: List[str]
    added_lines: List[ChangedLine]


@dataclass(frozen=True)
class DiffLine:
    kind: str  # "+" added, "-" removed, " " context
    text: str
    old_line: Optional[int]
    new_line: Optional[int]

    @property
    def raw(self) -> str:
        return self.kind + self.text

    @property
    def numbered(self) -> str:
        """Render the line with its new-file number so models need not count lines."""
        number = "" if self.new_line is None else str(self.new_line)
        return "%s%4s | %s" % (self.kind, number, self.text)


@dataclass
class Hunk:
    path: str
    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine] = field(default_factory=list)


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_META_PREFIXES = (
    "diff --git ", "index ", "new file mode", "deleted file mode", "old mode",
    "new mode", "similarity index", "dissimilarity index", "rename from",
    "rename to", "copy from", "copy to", "Binary files", "GIT binary patch",
)


def _header_path(raw: str) -> str:
    value = raw[4:].split("\t", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value[2:] if value.startswith(("a/", "b/")) else value


def _git_header_path(raw: str) -> str:
    rest = raw[len("diff --git "):]
    return rest.rsplit(" b/", 1)[-1].strip() if " b/" in rest else ""


def scan_diff(diff: str) -> Tuple[List[str], List[Hunk]]:
    """Parse a unified diff into changed files and hunks with exact line numbers.

    Inside a hunk the header counts decide where the body ends, so removed lines
    such as ``-- comment`` (rendered ``--- comment``) or added lines such as
    ``++i`` are never mistaken for file headers. Diffs whose counts are wrong are
    still read leniently: change lines that follow an exhausted hunk extend it
    until the next real header.
    """
    lines = diff.splitlines()
    files: List[str] = []
    hunks: List[Hunk] = []
    path = ""
    current: Optional[Hunk] = None
    old_left = new_left = old_no = new_no = 0
    for index, raw in enumerate(lines):
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file" never consumes hunk counts
        counted = current is not None and (old_left > 0 or new_left > 0)
        if not counted:
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if raw.startswith("--- ") and following.startswith("+++ "):
                current = None
                continue
            if raw.startswith("+++ "):
                path = _header_path(raw)
                if path != "/dev/null" and path not in files:
                    files.append(path)
                current = None
                continue
            if raw.startswith(_META_PREFIXES):
                if raw.startswith("diff --git "):
                    path = _git_header_path(raw) or path
                current = None
                continue
            match = HUNK.match(raw)
            if match:
                old_start, new_start = int(match.group(1)), int(match.group(3))
                old_left = int(match.group(2)) if match.group(2) is not None else 1
                new_left = int(match.group(4)) if match.group(4) is not None else 1
                current = Hunk(
                    path or "unknown", raw, old_start, old_left, new_start, new_left,
                )
                hunks.append(current)
                old_no, new_no = old_start, new_start
                continue
            if current is None or not raw or raw[0] not in " +-":
                current = None
                continue
        kind = raw[0] if raw else " "
        text = raw[1:]
        if kind == "+":
            current.lines.append(DiffLine("+", text, None, new_no))
            new_no += 1
            new_left -= 1
        elif kind == "-":
            current.lines.append(DiffLine("-", text, old_no, None))
            old_no += 1
            old_left -= 1
        elif kind == " ":
            current.lines.append(DiffLine(" ", text, old_no, new_no))
            old_no += 1
            new_no += 1
            old_left -= 1
            new_left -= 1
        else:
            current = None
    for hunk in hunks:
        if hunk.path not in files and hunk.path not in {"/dev/null", "unknown"}:
            files.append(hunk.path)
    return files, hunks


def parse_hunks(diff: str) -> List[Hunk]:
    return scan_diff(diff)[1]


def parse_unified_diff(diff: str) -> ParsedDiff:
    files, hunks = scan_diff(diff)
    added = [
        ChangedLine(hunk.path, line.new_line, line.text)
        for hunk in hunks for line in hunk.lines if line.kind == "+"
    ]
    return ParsedDiff(files=files, added_lines=added)
