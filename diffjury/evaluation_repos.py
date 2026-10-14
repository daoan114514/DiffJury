"""Throwaway checkouts of a reviewed revision for evaluation replays.

Real-PR datasets (e.g. SWR-Bench) review a diff *with* project context, so the
repository tools need the tree as it stood at the PR head. Clones are blobless
(``--filter=blob:none``); ``git worktree add`` batch-fetches the blobs one tree
needs, and the worktree is removed as soon as the review finishes.
"""
import os
import shutil
import subprocess
import threading
import uuid
from contextlib import contextmanager
from typing import Dict, Iterator


class SnapshotError(RuntimeError):
    pass


class GitSnapshots:
    def __init__(self, repos_dir: str, work_dir: str):
        self.repos_dir = os.path.abspath(repos_dir)
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, repository: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(repository, threading.Lock())

    def _git(self, repo: str, *args: str) -> None:
        options = []
        while args and args[0] == "-c":
            options += list(args[:2])
            args = args[2:]
        result = subprocess.run(
            ["git", "-c", "safe.directory=" + os.path.realpath(repo).replace("\\", "/"),
             *options, "-C", repo, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise SnapshotError("git %s: %s" % (" ".join(args[:2]), result.stderr.strip()[:300]))

    @contextmanager
    def checkout(self, repository: str, commit: str) -> Iterator[str]:
        repo = os.path.join(self.repos_dir, repository.replace("/", "__"))
        if os.path.commonpath([os.path.realpath(repo), self.repos_dir]) != self.repos_dir:
            raise SnapshotError("repository path escapes the local clone directory")
        if not os.path.isdir(repo):
            raise SnapshotError("no local clone for %s" % repository)
        # Short names: deep project trees plus a long base path can exceed
        # Windows' 260-character limit.
        target = os.path.join(self.work_dir, uuid.uuid4().hex[:8])
        if os.path.commonpath([os.path.realpath(target), self.work_dir]) != self.work_dir:
            raise SnapshotError("worktree path escapes its configured directory")
        # Worktree bookkeeping lives in the shared .git directory; serialise it per repo.
        with self._lock(repository):
            self._git(repo, "-c", "core.longpaths=true", "worktree", "add", "--detach", "--force", target, commit)
        try:
            yield target
        finally:
            with self._lock(repository):
                try:
                    self._git(repo, "-c", "core.longpaths=true", "worktree", "remove", "--force", target)
                except SnapshotError:
                    shutil.rmtree(target, ignore_errors=True)
                    self._git(repo, "worktree", "prune")


class SnapshotReviewer:
    """Wrap a case reviewer so each case is reviewed inside its own checkout."""

    def __init__(self, delegate, snapshots: GitSnapshots):
        self.delegate = delegate
        self.snapshots = snapshots
        self.name = getattr(delegate, "name", "snapshot-reviewer")

    def review(self, diff, parsed):
        return self.delegate.review(diff, parsed)

    def review_case(self, case, parsed):
        return self.review_case_detailed(case, parsed)[0]

    def review_case_detailed(self, case, parsed):
        commit = case.get("head_commit")
        if not commit:
            return self.delegate.review_case_detailed(case, parsed)
        with self.snapshots.checkout(case["repository"], commit) as root:
            return self.delegate.review_case_detailed({**case, "repository_root": root}, parsed)
