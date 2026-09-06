#!/usr/bin/env python3
"""Skill Manager Tool — agent-managed skill creation & editing.

Skills are the agent's procedural memory (narrow "how to do X"; MEMORY.md/USER.md are
broad, declarative). New skills land in ~/.hermes/skills/ (or ``skills.create_dir``);
existing skills (bundled, hub, user) are modified in place. Layout:
``<skills>/[category/]<skill>/SKILL.md`` + optional ``references/ templates/ scripts/ assets/``.
"""

import contextvars as _ctxvars
import hashlib
import json
import os
import stat
from contextlib import contextmanager, suppress
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home, display_hermes_home
from utils import atomic_write_text, is_truthy_value
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    parse_frontmatter as _parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT)
from tools.skill_manager_guards import (
    _background_review_preflight, _background_review_read_before_write_guard, _background_review_write_guard,
    _containing_skills_root, _curator_consolidation_delete_guard, _maybe_auto_propose_org_edit,
    _org_mirror_write_guard, _pinned_guard, _validate_delete_target, _is_background_review, _refusal as _err)
from tools.skill_manager_batch import _skill_manage_batch
from tools.skills_guard import scan_skill, scan_skill_content, should_allow_install, format_scan_report

def _reset_background_review_read_marks() -> None:
    """Reset shared review marks in the upstream guard sibling."""
    from tools.skill_manager_guards import _reset_background_review_read_marks as reset
    reset()

logger = logging.getLogger(__name__)

# Approval replay state stays task-local: a staging/replay operation must never
# authorize a sibling request in a concurrent gateway process.
_background_review_read_paths: "_ctxvars.ContextVar[Optional[_BackgroundReviewReadMarks]]" = (
    _ctxvars.ContextVar("background_review_read_paths", default=None)
)
_skill_gate_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False
)
_pending_apply_read_guard_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "pending_apply_read_guard_bypass", default=False
)
_MAX_PENDING_PRE_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_PENDING_PRE_IMAGE_ENTRIES = 16_384
_pending_apply_pre_image_hash: "_ctxvars.ContextVar[Optional[str]]" = _ctxvars.ContextVar(
    "pending_apply_pre_image_hash", default=None
)
_pending_target_anchor: "_ctxvars.ContextVar[Optional[Dict[str, Any]]]" = _ctxvars.ContextVar(
    "pending_target_anchor", default=None
)
_skill_write_thread_lock = threading.RLock()
_GUARD_AVAILABLE = True


class _BackgroundReviewReadMarks:
    """Read marks shared by copied tool contexts within one review run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._digests: Dict[str, str] = {}

    def add(self, path: str, digest: str) -> None:
        with self._lock:
            self._digests[path] = digest

    def contains(self, path: str, digest: str) -> bool:
        with self._lock:
            return self._digests.get(path) == digest

def mark_background_review_skill_read(
    path: Path, *, content_digest: Optional[str] = None
) -> None:
    """Compatibility facade for the upstream guard sibling's exact-byte mark."""
    from tools.skill_manager_guards import mark_background_review_skill_read as mark
    mark(path, content_digest=content_digest)


def _background_review_has_read(path: Path) -> bool:
    from tools.skill_manager_guards import _background_review_has_read
    return _background_review_has_read(path)

def _guard_agent_created_enabled_readonly() -> bool:
    """Read the create-scan policy without creating the Hermes directory tree."""
    try:
        from hermes_cli.config import read_raw_config, _expand_env_vars
        from hermes_cli.managed_scope import apply_managed_overlay

        cfg = apply_managed_overlay(_expand_env_vars(read_raw_config()))
        return is_truthy_value(
            cfg_get(cfg, "skills", "guard_agent_created"),
            default=False,
        )
    except Exception:
        return False

def _security_scan_new_skill_content(name: str, content: str) -> Optional[str]:
    """Scan exact proposed SKILL.md bytes before they become discoverable."""
    guard_enabled = None
    if _pending_target_anchor.get() is not None:
        # Approval replay must not let load_config() materialize an absent
        # skills root after the staged pre-image was bound.
        guard_enabled = _guard_agent_created_enabled_readonly()
    if guard_enabled is None:
        guard_enabled = _guard_agent_created_enabled()
    if not guard_enabled:
        return None
    if not _GUARD_AVAILABLE:
        return "Security scan failed closed: scanner is unavailable."
    try:
        result = scan_skill_content(
            content,
            skill_name=name,
            source="agent-created",
        )
        allowed, reason = should_allow_install(result)
        if allowed is not True:
            report = format_scan_report(result)
            return f"Security scan blocked this skill ({reason}):\n{report}"
    except Exception as e:
        logger.warning("Security scan failed for proposed skill %s: %s", name, e, exc_info=True)
        return "Security scan failed closed; the proposed skill was rejected."
    return None

def _background_review_read_before_write_guard(
    name: str,
    target: Path,
    action: str,
    file_label: str,
) -> Optional[Dict[str, Any]]:
    """Require review forks to load the exact target before mutating it."""
    if _pending_apply_read_guard_bypass.get():
        return None
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    if _background_review_has_read(target):
        return None

    return {
        "success": False,
        "error": (
            f"Refusing background curator {action} for skill '{name}': "
            f"the current {file_label} content has not been loaded in this "
            "review turn. Call skill_view(name) for SKILL.md, or "
            "skill_view(name, file_path=...) for a supporting file, then "
            "retry the write using the content just returned."
        ),
        "_read_before_write_required": True,
    }

def _background_review_staging_read_preflight(
    action: str,
    name: str,
    file_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Verify and bind a background review's exact read before staging."""
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None, False
    except Exception:
        return None, False

    existing = _find_skill(name)
    if not existing:
        return None, False
    if action == "edit" or (action == "patch" and not file_path):
        target = existing["path"] / "SKILL.md"
        file_label = "SKILL.md"
    elif action in {"patch", "write_file", "remove_file"}:
        if not file_path:
            return None, False
        target, error = _resolve_skill_target(existing["path"], file_path)
        if error or target is None or not target.exists():
            return None, False
        file_label = file_path or "SKILL.md"
    else:
        return None, False

    guard = _background_review_read_before_write_guard(
        name, target, action, file_label
    )
    return guard, guard is None

@contextmanager
def _skill_write_lock():
    """Serialize skill mutations across threads and POSIX Hermes processes."""
    with _skill_write_thread_lock:
        if os.name == "nt":
            yield
            return
        import fcntl

        lock_path = get_hermes_home() / ".skill-write.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                raise ValueError("skill write lock is not owner-only")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

def _pending_fs_identity(info: os.stat_result) -> tuple:
    """Bind replay identity to ctime so mtime-restored writes still diverge."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_mode,
    )

def _pending_refresh_directory_identity(
    anchor: Dict[str, Any], directory_fd: int, relative_key: str
) -> None:
    """Refresh metadata changed by Hermes' own descriptor-relative mutation."""
    info = os.fstat(directory_fd)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("pending skill target parent is not a directory")
    anchor.setdefault("tree_identities", {})[relative_key] = _pending_fs_identity(
        info
    )

def _hash_skill_tree_fd(
    root: Path,
    root_fd: int,
    identities: Optional[Dict[str, tuple]] = None,
) -> str:
    """Hash one skill tree through held no-follow directory descriptors."""
    root_info = os.fstat(root_fd)
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("skill target is not a real directory")
    if identities is not None:
        identities["."] = _pending_fs_identity(root_info)
    digest = hashlib.sha256()
    digest.update(
        (
            f"present\0{root.absolute()}\0{root_info.st_dev}\0{root_info.st_ino}\0"
            f"{stat.S_IMODE(root_info.st_mode):o}\0"
        ).encode("utf-8")
    )
    total_bytes = 0
    total_entries = 0

    def _walk(directory_fd: int, prefix: str = "") -> None:
        nonlocal total_bytes, total_entries
        directory_before = os.fstat(directory_fd)
        names = []
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                total_entries += 1
                if total_entries > _MAX_PENDING_PRE_IMAGE_ENTRIES:
                    raise ValueError("skill target tree exceeds entry limit")
        names.sort()
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("skill target tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        raise ValueError("skill target changed while hashing")
                    if identities is not None:
                        identities[relative] = _pending_fs_identity(opened)
                    digest.update(f"dir\0{relative}\0{mode:o}\0".encode("utf-8"))
                    _walk(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("skill target contains an unsupported file type")
            if info.st_nlink != 1:
                raise ValueError("skill target tree contains a hard-linked file")
            total_bytes += info.st_size
            if total_bytes > _MAX_PENDING_PRE_IMAGE_BYTES:
                raise ValueError("skill target tree exceeds the pre-image size limit")
            file_fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            try:
                before = os.fstat(file_fd)
                if (
                    (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino)
                    or before.st_nlink != 1
                    or not stat.S_ISREG(before.st_mode)
                ):
                    raise ValueError("skill target file changed before hashing")
                file_digest = hashlib.sha256()
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    file_digest.update(chunk)
                after = os.fstat(file_fd)
            finally:
                os.close(file_fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_nlink",
                "st_mode",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise ValueError("skill target changed while its pre-image was hashed")
            if identities is not None:
                identities[relative] = _pending_fs_identity(after)
            digest.update(b"file\0")
            digest.update(relative.encode("utf-8"))
            digest.update(f"\0{mode:o}\0{after.st_size}\0".encode("ascii"))
            digest.update(file_digest.digest())
            digest.update(b"\0")
        directory_after = os.fstat(directory_fd)
        directory_fields = (
            "st_dev",
            "st_ino",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        if any(
            getattr(directory_before, field) != getattr(directory_after, field)
            for field in directory_fields
        ):
            raise ValueError("skill target changed while its pre-image was hashed")

    _walk(root_fd)
    return digest.hexdigest()

def _target_tree_pre_image_hash(name: str, category: Optional[str] = None) -> str:
    """Hash a skill tree and the identity of the selected target directory."""
    found = _find_skill(name)
    digest = hashlib.sha256()
    if not found:
        destination = _resolve_skill_dir(name, category)
        skills_root = _skills_dir()
        digest.update(b"absent\0")
        digest.update(str(destination.resolve(strict=False)).encode("utf-8"))
        digest.update(b"\0")
        try:
            root_info = skills_root.lstat()
        except FileNotFoundError:
            anchor_root = skills_root.parent
            anchor_info = anchor_root.lstat()
            if stat.S_ISLNK(anchor_info.st_mode) or not stat.S_ISDIR(
                anchor_info.st_mode
            ):
                raise ValueError("skills root parent is not a real directory")
            digest.update(
                (
                    f"root-parent\0{anchor_root.resolve()}\0{anchor_info.st_dev}\0"
                    f"{anchor_info.st_ino}\0{stat.S_IMODE(anchor_info.st_mode):o}\0"
                ).encode("utf-8")
            )
            current_parent = skills_root
            digest.update(
                f"parent-absent\0{current_parent}\0".encode("utf-8")
            )
            relative_parent = destination.parent.relative_to(skills_root)
            for part in relative_parent.parts:
                current_parent = current_parent / part
                digest.update(
                    f"parent-absent\0{current_parent}\0".encode("utf-8")
                )
        else:
            if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
                raise ValueError("skills root is not a real directory")
            digest.update(
                (
                    f"root\0{skills_root.resolve()}\0{root_info.st_dev}\0"
                    f"{root_info.st_ino}\0{stat.S_IMODE(root_info.st_mode):o}\0"
                ).encode("utf-8")
            )
            try:
                relative_parent = destination.parent.relative_to(skills_root)
            except ValueError as exc:
                raise ValueError("new skill target escapes the skills root") from exc
            current_parent = skills_root
            missing_parent = False
            for part in relative_parent.parts:
                current_parent = current_parent / part
                if missing_parent:
                    digest.update(
                        f"parent-absent\0{current_parent}\0".encode("utf-8")
                    )
                    continue
                try:
                    parent_info = current_parent.lstat()
                except FileNotFoundError:
                    missing_parent = True
                    digest.update(
                        f"parent-absent\0{current_parent}\0".encode("utf-8")
                    )
                    continue
                if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(
                    parent_info.st_mode
                ):
                    raise ValueError("new skill target parent is not a real directory")
                digest.update(
                    (
                        f"parent\0{current_parent.resolve()}\0{parent_info.st_dev}\0"
                        f"{parent_info.st_ino}\0{stat.S_IMODE(parent_info.st_mode):o}\0"
                    ).encode("utf-8")
                )
        return digest.hexdigest()

    root = Path(found["path"])
    if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("descriptor-safe skill pre-image hashing is unsupported")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(root.parent, flags)
    try:
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            path_info = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(root_fd)
            if (
                not stat.S_ISDIR(path_info.st_mode)
                or (path_info.st_dev, path_info.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise ValueError("skill target identity changed while opening")
            result = _hash_skill_tree_fd(root, root_fd)
            final_path = os.stat(
                root.name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (final_path.st_dev, final_path.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ValueError("skill target identity changed while hashing")
            return result
        finally:
            os.close(root_fd)
    finally:
        os.close(parent_fd)

def _pending_pre_image_guard(
    name: str, category: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    expected = _pending_apply_pre_image_hash.get()
    if not expected:
        return None
    try:
        current = _target_tree_pre_image_hash(name, category)
    except (OSError, ValueError) as exc:
        return {
            "success": False,
            "error": f"Could not validate skill target pre-image: {exc}",
        }
    if current != expected:
        return {
            "success": False,
            "error": (
                "Pending skill write target pre-image changed; "
                "restage against the current skill."
            ),
        }
    return None

@contextmanager
def _pending_target_anchor_context(name: str, category: Optional[str] = None):
    """Bind approved writes to a held no-follow target-tree descriptor."""
    expected = _pending_apply_pre_image_hash.get()
    if not expected:
        yield None
        return
    if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("descriptor-anchored pending skill apply is unsupported")

    found = _find_skill(name)
    if found:
        root = Path(found["path"])
    else:
        skills_root = _skills_dir()
        try:
            skills_root_info = skills_root.lstat()
        except FileNotFoundError:
            root = skills_root.parent
        else:
            if stat.S_ISLNK(skills_root_info.st_mode) or not stat.S_ISDIR(
                skills_root_info.st_mode
            ):
                raise ValueError("skills root is not a real directory")
            root = skills_root
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(root.parent, flags)
    try:
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    token = None
    try:
        opened = os.fstat(root_fd)
        current_path = root.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current_path.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (current_path.st_dev, current_path.st_ino)
        ):
            raise ValueError("skill target identity changed before apply")
        tree_identities: Dict[str, tuple] = {".": _pending_fs_identity(opened)}
        if found:
            current_hash = _hash_skill_tree_fd(root, root_fd, tree_identities)
        else:
            current_hash = _target_tree_pre_image_hash(name, category)
            destination = _resolve_skill_dir(name, category)
            relative_parent = destination.parent.relative_to(root)
            chain_fd = os.dup(root_fd)
            prefix_parts: List[str] = []
            try:
                for part in relative_parent.parts:
                    try:
                        next_fd = os.open(
                            part,
                            flags,
                            dir_fd=chain_fd,
                        )
                    except FileNotFoundError:
                        break
                    prefix_parts.append(part)
                    tree_identities["/".join(prefix_parts)] = _pending_fs_identity(
                        os.fstat(next_fd)
                    )
                    os.close(chain_fd)
                    chain_fd = next_fd
            finally:
                os.close(chain_fd)
        current_path = root.lstat()
        if (
            current_hash != expected
            or (opened.st_dev, opened.st_ino)
            != (current_path.st_dev, current_path.st_ino)
        ):
            raise ValueError("Pending skill write target pre-image changed")
        anchor = {
            "root_fd": root_fd,
            "parent_fd": parent_fd,
            "root_name": root.name,
            "root_path": root,
            "root_dev": opened.st_dev,
            "root_ino": opened.st_ino,
            "target_exists": bool(found),
            "tree_identities": tree_identities,
        }
        token = _pending_target_anchor.set(anchor)
        yield anchor
    finally:
        if token is not None:
            _pending_target_anchor.reset(token)
        os.close(root_fd)
        os.close(parent_fd)

def _pending_anchor_is_current() -> bool:
    anchor = _pending_target_anchor.get()
    if anchor is None:
        return True
    try:
        current = Path(anchor["root_path"]).lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and (current.st_dev, current.st_ino)
        == (anchor["root_dev"], anchor["root_ino"])
    )

def _pending_assert_anchor_tree_current(
    expected_hash: Optional[str] = None,
) -> Optional[str]:
    """Validate and hash the published tree around a pathname-based scan."""
    anchor = _pending_target_anchor.get()
    if anchor is None:
        return None
    if not _pending_anchor_is_current():
        raise ValueError("Pending skill write target pre-image changed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(".", flags, dir_fd=anchor["root_fd"])
    identities: Dict[str, tuple] = {}
    try:
        current_hash = _hash_skill_tree_fd(
            Path(anchor["root_path"]), root_fd, identities
        )
    finally:
        os.close(root_fd)
    if (
        identities != anchor.get("tree_identities", {})
        or not _pending_anchor_is_current()
        or (expected_hash is not None and current_hash != expected_hash)
    ):
        raise ValueError("Pending skill write target pre-image changed")
    return current_hash

def _pending_assert_published_text_current(target: Path, expected: str) -> None:
    """Permit rollback only while the leaf still contains Hermes-published bytes."""
    if _pending_target_anchor.get() is None:
        return
    try:
        current = _pending_read_text(target)
    except FileNotFoundError as exc:
        raise ValueError("Pending skill write target pre-image changed") from exc
    if current != expected:
        raise ValueError("Pending skill write target pre-image changed")

@contextmanager
def _pending_target_parent_fd(target: Path, *, create_parents: bool = False):
    anchor = _pending_target_anchor.get()
    if anchor is None:
        yield None, target.name
        return
    try:
        relative = target.relative_to(Path(anchor["root_path"]))
    except ValueError as exc:
        raise ValueError("pending skill target escaped its anchored root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("pending skill target path is invalid")

    fd = os.dup(anchor["root_fd"])
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_parts = relative.parts[:-1]
        prefix_parts: List[str] = []
        identities = anchor.get("tree_identities", {})
        for part in parent_parts:
            parent_identity_key = "/".join(prefix_parts) or "."
            prefix_parts.append(part)
            identity_key = "/".join(prefix_parts)
            expected_identity = identities.get(identity_key)
            created = False
            if create_parents:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    if expected_identity is None:
                        raise ValueError(
                            "Pending skill write target pre-image changed"
                        )
                else:
                    if expected_identity is not None:
                        raise ValueError(
                            "Pending skill write target pre-image changed"
                        )
                    created = True
                    _pending_refresh_directory_identity(
                        anchor, fd, parent_identity_key
                    )
            elif expected_identity is None:
                raise ValueError("Pending skill write target pre-image changed")
            next_fd = os.open(part, flags, dir_fd=fd)
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_fd)
                raise ValueError("pending skill target parent is not a directory")
            current_identity = _pending_fs_identity(info)
            if expected_identity is not None and current_identity != expected_identity:
                os.close(next_fd)
                raise ValueError("Pending skill write target pre-image changed")
            if created:
                identities[identity_key] = current_identity
            os.close(fd)
            fd = next_fd
        yield fd, relative.parts[-1]
    finally:
        os.close(fd)

def _pending_read_text(target: Path) -> str:
    anchor = _pending_target_anchor.get()
    if anchor is None:
        return target.read_text(encoding="utf-8")
    with _pending_target_parent_fd(target) as (parent_fd, leaf):
        assert parent_fd is not None
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("pending skill target file is unsafe")
            chunks = []
            total = 0
            while True:
                chunk = os.read(fd, min(1024 * 1024, MAX_SKILL_FILE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_SKILL_FILE_BYTES:
                    raise ValueError("pending skill target file exceeds size limit")
            after = os.fstat(fd)
            if _pending_fs_identity(before) != _pending_fs_identity(after):
                raise ValueError("pending skill target changed while reading")
            relative_key = target.relative_to(Path(anchor["root_path"])).as_posix()
            anchor.setdefault("read_identities", {})[
                relative_key
            ] = _pending_fs_identity(after)
            return b"".join(chunks).decode("utf-8")
        finally:
            os.close(fd)

def _atomic_create_text_noreplace(
    target: Path,
    content: str,
    *,
    create_parents: bool = False,
    create_mode: int = 0o644,
) -> None:
    """Atomically publish a new text leaf without replacing an existing one."""
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(
        f".pending-{os.getpid()}-{os.urandom(8).hex()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    linked = False
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, create_mode)
        view = memoryview(content.encode("utf-8"))
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("skill create staging write made no progress")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(temp, target)
        linked = True
        os.unlink(temp)
    finally:
        if fd >= 0:
            os.close(fd)
        if not linked or temp.exists():
            try:
                os.unlink(temp)
            except OSError:
                pass

def _pending_atomic_write_text(
    target: Path,
    content: str,
    *,
    create_parents: bool = False,
    expect_absent: bool = False,
) -> None:
    anchor = _pending_target_anchor.get()
    if anchor is None:
        if expect_absent:
            _atomic_create_text_noreplace(
                target,
                content,
                create_parents=create_parents,
                create_mode=0o644,
            )
        else:
            atomic_write_text(
                target,
                content,
                preserve_mode=True,
                create_mode=0o644,
            )
        return
    if not _pending_anchor_is_current():
        raise ValueError("Pending skill write target pre-image changed")
    with _pending_target_parent_fd(target, create_parents=create_parents) as (
        parent_fd,
        leaf,
    ):
        assert parent_fd is not None
        try:
            existing = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if expect_absent and existing is not None:
            raise ValueError("Pending skill write target pre-image changed")
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise ValueError("pending skill target file is unsafe")
        relative_key = target.relative_to(Path(anchor["root_path"])).as_posix()
        expected_identity = anchor.get("read_identities", {}).get(relative_key)
        if expected_identity is None:
            expected_identity = anchor.get("tree_identities", {}).get(relative_key)
        current_identity = (
            None if existing is None else _pending_fs_identity(existing)
        )
        if current_identity != expected_identity:
            raise ValueError("Pending skill write target pre-image changed")
        temp_name = f".pending-{os.getpid()}-{os.urandom(8).hex()}.tmp"
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        published = False
        try:
            publish_mode = (
                stat.S_IMODE(existing.st_mode) if existing is not None else 0o644
            )
            if hasattr(os, "fchmod"):
                os.fchmod(fd, publish_mode)
            if existing is not None:
                if not hasattr(os, "fchown"):
                    raise PermissionError(
                        "pending skill target ownership cannot be preserved"
                    )
                os.fchown(fd, existing.st_uid, existing.st_gid)
                temp_owner = os.fstat(fd)
                if (
                    temp_owner.st_uid != existing.st_uid
                    or temp_owner.st_gid != existing.st_gid
                ):
                    raise PermissionError(
                        "pending skill target ownership was not preserved"
                    )
            data = content.encode("utf-8")
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            if not _pending_anchor_is_current():
                raise ValueError("Pending skill write target pre-image changed")
            try:
                before_publish = os.stat(
                    leaf, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                before_publish_identity = None
            else:
                before_publish_identity = _pending_fs_identity(before_publish)
            if before_publish_identity != current_identity:
                raise ValueError("Pending skill write target pre-image changed")
            if expect_absent:
                os.link(
                    temp_name,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                os.unlink(temp_name, dir_fd=parent_fd)
            else:
                os.replace(
                    temp_name,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            published = True
            published_identity = _pending_fs_identity(
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            )
            anchor.setdefault("tree_identities", {})[
                relative_key
            ] = published_identity
            anchor.setdefault("read_identities", {})[
                relative_key
            ] = published_identity
            os.fsync(parent_fd)
            parent_relative = target.parent.relative_to(
                Path(anchor["root_path"])
            ).as_posix()
            _pending_refresh_directory_identity(
                anchor,
                parent_fd,
                "." if parent_relative == "." else parent_relative,
            )
        finally:
            if fd >= 0:
                os.close(fd)
            if not published:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except OSError:
                    pass

def _pending_unlink(target: Path) -> None:
    anchor = _pending_target_anchor.get()
    if anchor is None:
        target.unlink()
        return
    if not _pending_anchor_is_current():
        raise ValueError("Pending skill write target pre-image changed")
    with _pending_target_parent_fd(target) as (parent_fd, leaf):
        assert parent_fd is not None
        info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("pending skill target file is unsafe")
        relative_key = target.relative_to(Path(anchor["root_path"])).as_posix()
        expected_identity = anchor.get("tree_identities", {}).get(relative_key)
        if expected_identity is None or _pending_fs_identity(info) != expected_identity:
            raise ValueError("Pending skill write target pre-image changed")
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if _pending_fs_identity(current) != expected_identity:
            raise ValueError("Pending skill write target pre-image changed")
        os.unlink(leaf, dir_fd=parent_fd)
        anchor.get("tree_identities", {}).pop(relative_key, None)
        anchor.get("read_identities", {}).pop(relative_key, None)
        os.fsync(parent_fd)
        parent_relative = target.parent.relative_to(
            Path(anchor["root_path"])
        ).as_posix()
        _pending_refresh_directory_identity(
            anchor,
            parent_fd,
            "." if parent_relative == "." else parent_relative,
        )

def _snapshot_pending_delete_tree_fd(
    directory_fd: int,
    remaining: List[int],
    *,
    prefix: str = "",
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and snapshot a delete tree completely before any mutation."""
    result = snapshot if snapshot is not None else {"children": {}, "identities": {}}
    directory_before = os.fstat(directory_fd)
    names = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > remaining[0]:
                raise ValueError("pending skill target tree has too many entries")
    names.sort()
    remaining[0] -= len(names)
    result["children"][prefix] = names
    result["identities"][prefix or "."] = _pending_fs_identity(directory_before)
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("pending skill target tree contains a symbolic link")
        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("Pending skill write target pre-image changed")
                _snapshot_pending_delete_tree_fd(
                    child_fd,
                    remaining,
                    prefix=relative,
                    snapshot=result,
                )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("pending skill target tree contains an unsafe file")
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            opened = os.fstat(file_fd)
            if _pending_fs_identity(opened) != _pending_fs_identity(info):
                raise ValueError("Pending skill write target pre-image changed")
            result["identities"][relative] = _pending_fs_identity(opened)
        finally:
            os.close(file_fd)
    directory_after = os.fstat(directory_fd)
    if _pending_fs_identity(directory_after) != _pending_fs_identity(directory_before):
        raise ValueError("Pending skill write target pre-image changed")
    return result

def _pending_assert_snapshot_matches_anchor(
    target: Path, snapshot: Dict[str, Any]
) -> None:
    """Require a rollback tree to equal the state published by this replay."""
    anchor = _pending_target_anchor.get()
    if anchor is None:
        return
    try:
        prefix = target.relative_to(Path(anchor["root_path"])).as_posix()
    except ValueError as exc:
        raise ValueError("pending skill target escaped its anchored root") from exc
    identities = anchor.get("tree_identities", {})
    if prefix == ".":
        expected = dict(identities)
    else:
        expected = {}
        for key, identity in identities.items():
            if key == prefix:
                expected["."] = identity
            elif key.startswith(f"{prefix}/"):
                expected[key[len(prefix) + 1 :]] = identity
    if not expected or expected != snapshot.get("identities", {}):
        raise ValueError("Pending skill write target pre-image changed")

def _pending_quarantine_tree(parent_fd: int, leaf: str, target_fd: int) -> str:
    """Atomically move the exact approved tree into an excluded quarantine."""
    expected = _pending_fs_identity(os.fstat(target_fd))
    current_euid = getattr(os, "geteuid", lambda: None)()
    try:
        os.mkdir(".archive", 0o755, dir_fd=parent_fd)
    except FileExistsError:
        pass
    archive_fd = os.open(
        ".archive",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        archive_info = os.fstat(archive_fd)
        if (
            not stat.S_ISDIR(archive_info.st_mode)
            or stat.S_IMODE(archive_info.st_mode) & 0o022
            or (current_euid is not None and archive_info.st_uid != current_euid)
        ):
            raise ValueError("pending skill delete archive is unsafe")
        try:
            os.mkdir(".pending-deletes", 0o700, dir_fd=archive_fd)
        except FileExistsError:
            pass
        quarantine_fd = os.open(
            ".pending-deletes",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=archive_fd,
        )
        try:
            quarantine_info = os.fstat(quarantine_fd)
            if (
                not stat.S_ISDIR(quarantine_info.st_mode)
                or stat.S_IMODE(quarantine_info.st_mode) != 0o700
                or (
                    current_euid is not None
                    and quarantine_info.st_uid != current_euid
                )
            ):
                raise ValueError("pending skill delete quarantine is unsafe")
            tombstone = f"{leaf}-{os.urandom(12).hex()}"
            os.rename(
                leaf,
                tombstone,
                src_dir_fd=parent_fd,
                dst_dir_fd=quarantine_fd,
            )
            moved = os.stat(
                tombstone,
                dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
            if _pending_fs_identity(moved) != expected:
                try:
                    os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    os.rename(
                        tombstone,
                        leaf,
                        src_dir_fd=quarantine_fd,
                        dst_dir_fd=parent_fd,
                    )
                raise ValueError("Pending skill write target pre-image changed")
            os.fsync(quarantine_fd)
            os.fsync(parent_fd)
            return tombstone
        finally:
            os.close(quarantine_fd)
    finally:
        os.close(archive_fd)

def _pending_rmtree(target: Path) -> None:
    anchor = _pending_target_anchor.get()
    if anchor is None:
        shutil.rmtree(target)
        return
    if not _pending_anchor_is_current():
        raise ValueError("Pending skill write target pre-image changed")

    def _remove_at(parent_fd: int, leaf: str, target_fd: int) -> None:
        opened = os.fstat(target_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError("pending skill target directory is unsafe")
        snapshot = _snapshot_pending_delete_tree_fd(
            target_fd,
            [_MAX_PENDING_PRE_IMAGE_ENTRIES],
        )
        _pending_assert_snapshot_matches_anchor(target, snapshot)
        _pending_quarantine_tree(parent_fd, leaf, target_fd)

    if target == Path(anchor["root_path"]):
        parent_fd = os.dup(anchor["parent_fd"])
        target_fd = os.dup(anchor["root_fd"])
        try:
            _remove_at(parent_fd, anchor["root_name"], target_fd)
        finally:
            os.close(target_fd)
            os.close(parent_fd)
        return

    with _pending_target_parent_fd(target) as (parent_fd, leaf):
        assert parent_fd is not None
        target_fd = os.open(
            leaf,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            _remove_at(parent_fd, leaf, target_fd)
        finally:
            os.close(target_fd)

def _skill_manage_unlocked(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    operations=None,
) -> str:
    """
    Manage user-created skills. Dispatches to the appropriate action handler.

    ``operations``: batch shape — a list of {action, ...} dicts applied to
    ONE skill atomically (see _skill_manage_batch). When set, the flat
    single-op fields are ignored and ``action`` may be omitted/'batch'.

    Returns JSON string with results.
    """
    if operations is not None:
        return _skill_manage_batch(
            operations,
            default_name=name or None,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
        )
    preflight = _background_review_preflight(action, name)
    if preflight is not None:
        return json.dumps(preflight, ensure_ascii=False)

    read_preflight, background_review_read_verified = (
        _background_review_staging_read_preflight(action, name, file_path)
    )
    if read_preflight is not None:
        return json.dumps(read_preflight, ensure_ascii=False)

    # Approval gate: when on, stages the write for review (skills are too large
    # to review inline, so they always stage regardless of origin); when off
    # (default) passes straight through. The gate is bypassed when this call is
    # itself replaying an already-approved staged write (_skill_apply_pending).
    gate_result = _apply_skill_write_gate(
        action, name, content=content, category=category,
        file_path=file_path, file_content=file_content,
        old_string=old_string, new_string=new_string,
        replace_all=replace_all, absorbed_into=absorbed_into,
        session_id=session_id, tool_call_id=tool_call_id,
        background_review_read_verified=background_review_read_verified,
    )
    if gate_result is not None:
        return gate_result

    # Audit ledger (tracker #79686 P3): capture the pre-mutation state of the
    # skill directory so every mutation — any actor — lands in the append-only
    # JSONL ledger with before/after blobs. Telemetry, not a gate: failures
    # here must NEVER block the mutation (capture_before returns None on
    # error, and record_mutation below swallows everything).
    _ledger_before = None
    _ledger_before_dir = None
    try:
        from tools import skill_ledger as _ledger
        _pre = _find_skill(name)
        _ledger_before_dir = _pre["path"] if _pre else None
        # delete destroys the whole package; consolidation may have re-homed
        # support files out of the tree first, so complete the capture from
        # the newest curator backup or rollback restores a hollow skill
        # (#96962). Other actions capture disk state only.
        _ledger_before = (
            []
            if _ledger_before_dir is None
            else _ledger.capture_before(
                _ledger_before_dir,
                complete_package=(action == "delete"),
                skill=name,
            )
        )
    except Exception:
        pass

    if action == "create":
        if not content:
            return tool_error("content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).", success=False)
        result = _create_skill(name, content, category)

    elif action == "edit":
        # Legacy alias for a full rewrite (kept for old transcripts/callers;
        # no longer advertised in the schema — use patch with `content`).
        if not content:
            return tool_error("content is required for a full rewrite. Provide the full updated SKILL.md text.", success=False)
        result = _edit_skill(name, content)

    elif action == "patch":
        # Two shapes: old_string/new_string = targeted replacement;
        # content (alone) = full SKILL.md rewrite (absorbs the old 'edit').
        if content and (old_string or new_string is not None):
            return tool_error(
                "Pass EITHER content (full SKILL.md rewrite) OR "
                "old_string/new_string (targeted replacement), not both.",
                success=False,
            )
        if content:
            result = _edit_skill(name, content)
        else:
            # Targeted-replacement validation lives in _patch_skill so the
            # public tool and the helper return the same actionable guidance.
            # A bare "required" error here would shadow it and leave the
            # model with nowhere to go but action='write_file'. #33064.
            result = _patch_skill(name, old_string, new_string, file_path, replace_all)

    elif action == "delete":
        result = _delete_skill(name, absorbed_into=absorbed_into)

    elif action == "write_file":
        if not file_path:
            return tool_error("file_path is required for 'write_file'. Example: 'references/api-guide.md'", success=False)
        if file_content is None:
            return tool_error("file_content is required for 'write_file'.", success=False)
        result = _write_file(name, file_path, file_content)

    elif action == "remove_file":
        if not file_path:
            return tool_error("file_path is required for 'remove_file'.", success=False)
        result = _remove_file(name, file_path)

    else:
        result = {"success": False, "error": f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file"}

    if result.get("success"):
        # Audit ledger append (best-effort; never blocks the mutation).
        try:
            from tools import skill_ledger as _ledger
            _post = _find_skill(name)
            _after_dir = _post["path"] if _post else None
            _evidence = {}
            if action == "delete":
                # Record delete intent: consolidation vs prune, and whether
                # the recoverable-archive path handled it (curator pass).
                _evidence["absorbed_into"] = absorbed_into
                _evidence["archived"] = bool(result.get("_archived"))
            if session_id:
                _evidence["session_id"] = session_id
            if file_path:
                _evidence["file_path"] = file_path
            _ledger.record_mutation(
                action,
                name,
                before=_ledger_before if _ledger_before is not None else [],
                after_root=_after_dir,
                evidence=_evidence,
            )
        except Exception:
            pass
        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache
            clear_skills_system_prompt_cache(clear_snapshot=True)
        except Exception:
            pass
        # Curator telemetry: bump patch_count on edit/patch/write_file (the actions
        # that mutate an existing skill's guidance), drop the record on delete.
        # Only mark a skill as agent-created when the background self-improvement
        # review fork creates it — foreground `skill_manage(create)` calls are
        # user-directed, and those skills belong to the user (the curator must
        # not touch them). Best-effort; telemetry failures never break the tool.
        try:
            from tools.skill_usage import bump_patch, forget, record_created
            from tools.skill_provenance import is_background_review
            if action == "create":
                record_created(
                    name,
                    agent_created=is_background_review(),
                    task_id=task_id,
                    session_id=session_id,
                )
            elif action in {"patch", "edit", "write_file", "remove_file"}:
                bump_patch(
                    name,
                    action=action,
                    task_id=task_id,
                    session_id=session_id,
                )
            elif action == "delete":
                # A recoverable curator archive (routed through archive_skill)
                # keeps its usage record as STATE_ARCHIVED so `hermes curator
                # status`/`restore` still see it. Only a hard delete forgets.
                if not result.get("_archived"):
                    forget(name)
        except Exception:
            pass

        # Sync push hook (debounced, best-effort). Fires only AFTER the
        # write gate passed (staged/unapproved writes never reach here -- the
        # gate returns early above), so we never push un-reviewed content.
        # Inert unless the access gate is open (the user is a Nous admin on the
        # token), a sync base URL is configured, and the skill is opted into
        # sync. Debounced so a burst of edits collapses to one push. Never
        # raises -- an agent write must never block on sync (M1-C invariant).
        try:
            _maybe_debounced_sync_push(name)
        except Exception:
            pass

    return json.dumps(result, ensure_ascii=False)


def _guard_agent_created_enabled() -> bool:
    """skills.guard_agent_created (default False): opt-in — terminal() runs the same code ungated."""
    try:
        from hermes_cli.config import load_config
        return is_truthy_value(cfg_get(load_config(), "skills", "guard_agent_created"), default=False)
    except Exception:
        return False


def _security_scan_skill(
    skill_dir: Path, *, guard_enabled: Optional[bool] = None
) -> Optional[str]:
    """Scan a skill directory after write. Returns error string if blocked, else None.

    No-op when skills.guard_agent_created is disabled (the default).
    """
    if guard_enabled is None:
        guard_enabled = _guard_agent_created_enabled()
    if not guard_enabled:
        return None
    if not _GUARD_AVAILABLE:
        return "Security scan failed closed: scanner is unavailable."
    try:
        result = scan_skill(skill_dir, source="agent-created")
        allowed, reason = should_allow_install(result)
        if allowed is False:
            report = format_scan_report(result)
            return f"Security scan blocked this skill ({reason}):\n{report}"
        if allowed is None:
            # "ask" verdict — for agent-created skills this means dangerous
            # findings were detected.  Surface as an error so the agent can
            # retry with the flagged content removed.
            report = format_scan_report(result)
            logger.warning("Agent-created skill blocked (dangerous findings): %s", reason)
            return f"Security scan blocked this skill ({reason}):\n{report}"
    except Exception as e:
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
        return "Security scan failed closed; the skill write was rejected."
    return None

# All skills live in ~/.hermes/skills/ (single source of truth)
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time (multi-profile runtimes rebind per session).
    An explicitly patched module-level ``SKILLS_DIR`` (tests) wins over the live HERMES_HOME.

    Long-lived multi-profile runtimes (Dashboard/TUI/Desktop backend, cron, kanban workers) import this
    module once under the launch HERMES_HOME and later bind a different profile per session (#40677).
    """
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000   # ~36k tokens at 2.75 chars/token
MAX_SKILL_FILE_BYTES = 1_048_576    # 1 MiB per supporting file
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')  # filesystem-safe, URL-friendly
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}  # for write_file/remove_file
_FRONTMATTER_END_RE = re.compile(r'\n---\s*\n')
_NAME_RULE = "Use lowercase letters, numbers, hyphens, dots, and underscores."


def _display_create_dir() -> str:
    """Skill-creation dir for schema/instruction text; follows ``skills.create_dir``."""
    try:
        from agent.skill_utils import display_skill_create_dir
        return display_skill_create_dir()
    except Exception:
        return f"{display_hermes_home()}/skills/"


# --- Validation helpers -------------------------------------------------------

def _check_identifier(value: str, label: str, invalid: str) -> Optional[str]:
    if len(value) > MAX_NAME_LENGTH:
        return f"{label} exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(value) else invalid


def _validate_name(name: str) -> Optional[str]:
    if not name:
        return "Skill name is required."
    return _check_identifier(
        name, "Skill name", f"Invalid skill name '{name}'. {_NAME_RULE} Must start with a letter or digit.")


def _validate_category(category: Optional[str]) -> Optional[str]:
    if category is None or (isinstance(category, str) and not category.strip()):
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    category = category.strip()
    invalid = (f"Invalid category '{category}'. {_NAME_RULE} "
               "Categories must be a single directory name.")
    if "/" in category or "\\" in category:
        return invalid
    return _check_identifier(category, "Category", invalid)


def _validate_frontmatter(content: str, *, new_skill: bool = False) -> Optional[str]:
    """Validate frontmatter (name + description) and a non-empty body. ``new_skill`` (create
    only) also enforces SKILL_PROMPT_DESC_LIMIT so new skills never lose routing signal to
    index truncation; edit/patch skip it so existing over-limit skills stay maintainable."""
    if not content.strip():
        return "Content cannot be empty."
    content = content.lstrip("\ufeff")  # tolerate a Windows UTF-8 BOM
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."
    end_match = _FRONTMATTER_END_RE.search(content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."
    try:
        parsed = yaml.safe_load(content[3:end_match.start() + 3])
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"
    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."
    for field in ("name", "description"):
        if field not in parsed:
            return f"Frontmatter must include '{field}' field."
    desc = str(parsed["description"])
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if new_skill and len(desc.strip().strip("'\"")) > SKILL_PROMPT_DESC_LIMIT:
        return (
            f"Description is {len(desc.strip())} chars — new skills must fit the "
            f"{SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget (one sentence, trigger first, "
            f"ends with a period). The skill index truncates longer descriptions to "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars + '...', destroying the routing signal. "
            f"Move detail into the skill body.")
    if not content[end_match.end() + 3:].strip():
        return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."
    return None


def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        return (
            f"{label} content is {len(content):,} characters (limit: {MAX_SKILL_CONTENT_CHARS:,}). "
            f"Consider splitting into a smaller SKILL.md with supporting files in references/ "
            f"or templates/.")
    return None


def _description_preview(content: str) -> str:
    """First 120 chars of the frontmatter description; '' on any failure."""
    with suppress(Exception):
        fm_end = _FRONTMATTER_END_RE.search(content[3:])
        if fm_end:
            return str(yaml.safe_load(content[3:fm_end.start() + 3]).get("description", ""))[:120]
    return ""


def _resolve_skill_dir(name: str, category: str = None) -> Path:
    """New-skill dir; honors ``skills.create_dir`` (e.g. a shared fleet dir)."""
    base = _skills_dir()
    try:
        from agent.skill_utils import get_skill_create_dir
        base = get_skill_create_dir() or base
    except Exception:
        logger.debug("skills.create_dir lookup failed", exc_info=True)
    return base / (category or "") / name


def _iter_skill_dirs(root: Path):
    from agent.skill_utils import is_excluded_skill_path
    for skill_md in root.rglob("SKILL.md"):
        if not is_excluded_skill_path(skill_md):
            yield skill_md.parent


def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """Find a skill (local skills dir, then skills.external_dirs) -> ``{"path": Path}`` | None.

    Accepts the bare dir name (``axolotl``; matches category-nested skills too) and the
    categorized relative path (``mlops/axolotl``) — the two forms skill_view resolves. The
    categorized form matches RELATIVE to the local root only (relative_to raises for external dirs)."""
    from agent.skill_utils import get_all_skills_dirs
    local_root = None
    if "/" in name or "\\" in name:
        try:
            local_root = _skills_dir().resolve()
        except OSError:
            logger.debug(
                "skills dir resolve failed; categorized lookups fall back to the unresolved path",
                exc_info=True)
            local_root = _skills_dir()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_dir in _iter_skill_dirs(skills_dir):
            if skill_dir.name == name:
                return {"path": skill_dir}
            if local_root is not None:
                resolved = skill_dir.resolve()
                if (resolved.is_relative_to(local_root)
                        and resolved.relative_to(local_root).as_posix() == name):  # POSIX form
                    return {"path": skill_dir}
    return None


def _find_skill_in_other_profiles(name: str) -> List[Tuple[str, Path]]:
    """``(profile, skill_dir)`` pairs for OTHER profiles holding ``name`` (so the not-found
    error can explain a wrong-profile mistake). Fail-quiet."""
    matches: List[Tuple[str, Path]] = []
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
    except Exception:
        return matches
    _active = _skills_dir()
    active_dir = _active.resolve() if _active.exists() else _active
    # Every profile's skills dir EXCEPT the active one (already searched). A candidate whose
    # path cannot be resolved is skipped (not a fatal error); is_dir() checks stay unguarded.
    candidates: List[Tuple[str, Path]] = []
    with suppress(OSError, RuntimeError):
        if (root / "skills").resolve() != active_dir:
            candidates.append(("default", root / "skills"))
    if (root / "profiles").is_dir():
        with suppress(OSError):
            for entry in (root / "profiles").iterdir():
                if not entry.is_dir():
                    continue
                try:
                    if (entry / "skills").resolve() == active_dir:
                        continue
                except (OSError, RuntimeError):
                    continue
                candidates.append((entry.name, entry / "skills"))
    for profile_name, skills_dir in candidates:
        if not skills_dir.is_dir():
            continue
        with suppress(OSError):
            hit = next((d for d in _iter_skill_dirs(skills_dir) if d.name == name), None)
            if hit is not None:
                matches.append((profile_name, hit))  # one match per profile is enough
    return matches


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    """Not-found error naming other profiles that hold the skill, plus ``suffix``."""
    from agent.file_safety import _resolve_active_profile_name
    base = f"Skill '{name}' not found in active profile '{_resolve_active_profile_name()}'."
    others = _find_skill_in_other_profiles(name)
    if len(others) == 1:
        other_profile, other_path = others[0]
        base += (
            f" A skill by that name exists in profile '{other_profile}' ({other_path}). To edit "
            f"it, switch profiles (`hermes -p {other_profile}`) or edit the file directly "
            f"(file tools / terminal).")
    elif others:
        names = ", ".join(f"'{p}'" for p, _ in others)
        base += (
            f" Skills by that name exist in other profiles: {names}. Switch profiles (`hermes -p "
            f"<name>`) to edit there, or edit the files directly (file tools / terminal).")
    else:
        base += " Use skills_list() to see available skills."
    return base + suffix


def _resolve_skill_target(skill_dir: Path, file_path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a supporting-file path and ensure it stays within the skill directory."""
    from tools.path_security import validate_within_dir

    target = skill_dir / file_path
    error = validate_within_dir(target, skill_dir)
    if error:
        return None, error
    return target, None

def _validate_file_path(file_path: str) -> Optional[str]:
    """Validate a write_file/remove_file path: under an allowed subdir, no escape."""
    from tools.path_security import has_traversal_component
    if not file_path:
        return "file_path is required."
    parts = Path(file_path).parts
    # Traversal first, so the SKILL.md exception is unreachable by a traversal-laden path.
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."
    # SKILL.md lives at the skill root; accept 'SKILL.md' and '<skill>/SKILL.md'.
    if parts and parts[-1] == "SKILL.md" and len(parts) in (1, 2):
        return None
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"
    if len(parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{parts[0]}/myfile.md'"
    return None


def _resolve_supporting_file(skill_dir: Path, file_path: str):
    """Validate ``file_path`` and resolve it inside ``skill_dir``
    -> ``(target, None)`` | ``(None, error_dict)``."""
    from tools.path_security import validate_within_dir
    target = skill_dir / (file_path or "")
    err = _validate_file_path(file_path) or validate_within_dir(target, skill_dir)
    return (None, _err(err)) if err else (target, None)


def _locate_for_write(name: str, action: str, not_found_suffix: str = "", *,
                      org_guard: bool = True):
    """Find the skill; run the org-mirror (unless ``org_guard=False``) and background-review
    write guards -> ``(skill_dir, None)`` | ``(None, error_dict)``."""
    existing = _find_skill(name)
    if not existing:
        return None, _err(_skill_not_found_error(name, not_found_suffix))
    skill_dir = existing["path"]
    guard = ((org_guard and _org_mirror_write_guard(name, skill_dir, action))
             or _background_review_write_guard(name, skill_dir, action))
    return (None, guard) if guard else (skill_dir, None)


def _guarded_write(name: str, skill_dir: Path, target: Path, action: str, label: str,
                   content: str) -> Optional[Dict[str, Any]]:
    """Read-before-write guard (existing targets only), atomic write, then the security scan;
    a blocked scan restores the original (or unlinks a new file). Error dict or None."""
    original = None
    if target.exists():
        if read_guard := _background_review_read_before_write_guard(name, target, action, label):
            return read_guard
        original = target.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, content, preserve_mode=True, create_mode=0o644)
    scan_error = _security_scan_skill(skill_dir)
    if not scan_error:
        return None
    if original is not None:
        atomic_write_text(target, original, preserve_mode=True)
    else:
        target.unlink(missing_ok=True)
    return _err(scan_error)


def _attach_org_note(result: Dict[str, Any], name: str, skill_dir: Path) -> Dict[str, Any]:
    if org_note := _maybe_auto_propose_org_edit(name, skill_dir):
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    return result


def _add_description_prompt_preview(result: Dict[str, Any], content: str) -> Dict[str, Any]:
    fm, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(fm):
        result["system_prompt_preview"] = (
            f"System prompt will show: \"{extract_skill_description(fm)}\" — keep the trigger "
            f"self-contained in the first {SKILL_PROMPT_DESC_LIMIT - 3} chars.")
    return result


def _attach_lint_findings(result: Dict[str, Any], skill_md: Path) -> None:
    """Attach ADVISORY authoring findings (hard rejects already ran in _validate_frontmatter)."""
    try:
        from tools.skill_linter import lint_skill  # local import: optional path
        findings = lint_skill(skill_md)
    except Exception:
        findings = None
    if not findings:
        return
    result["lint_warnings"] = [
        {"severity": f.severity, "rule": f.rule, "message": f.message} for f in findings]
    result["lint_hint"] = (
        "The skill was created. These are advisory authoring-convention findings (not blockers) "
        "— fix them with skill_manage(action='patch') to match Hermes skill standards.")


def _clip(text: str, n: int, ellipsis: str) -> str:
    return text[:n] + (ellipsis if len(text) > n else "")


# --- Core actions -------------------------------------------------------------

def _create_skill(name: str, content: str, category: str = None) -> Dict[str, Any]:
    """Create a new user skill with SKILL.md content."""
    # Validate name
    err = _validate_name(name)
    if err:
        return {"success": False, "error": err}

    err = _validate_category(category)
    if err:
        return {"success": False, "error": err}

    # Validate content
    err = _validate_frontmatter(content, new_skill=True)
    if err:
        return {"success": False, "error": err}

    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    # Check for name collisions across all directories
    existing = _find_skill(name)
    if existing:
        return {
            "success": False,
            "error": f"A skill named '{name}' already exists at {existing['path']}.",
        }

    pre_image_guard = _pending_pre_image_guard(name, category)
    if pre_image_guard:
        return pre_image_guard

    # Scan the exact proposed bytes before publishing SKILL.md. A rejected
    # create never becomes discoverable and needs no race-prone rollback.
    scan_error = _security_scan_new_skill_content(name, content)
    if scan_error:
        return {"success": False, "error": scan_error}

    # Create the skill directory
    skill_dir = _resolve_skill_dir(name, category)
    if _pending_target_anchor.get() is None:
        skill_dir.mkdir(parents=True, exist_ok=True)

    # Write instructional documents with a readable mode while preserving
    # the mode of an existing file across the atomic replacement.
    skill_md = skill_dir / "SKILL.md"
    pre_image_guard = _pending_pre_image_guard(name, category)
    if pre_image_guard:
        return pre_image_guard
    _pending_atomic_write_text(
        skill_md, content, create_parents=True, expect_absent=True
    )

    # Extract description from frontmatter for verbose notifications
    _desc = ""
    try:
        _fm_end = re.search(r'\n---\s*\n', content[3:])
        if _fm_end:
            _parsed = yaml.safe_load(content[3:_fm_end.start() + 3])
            _desc = str(_parsed.get("description", ""))[:120]
    except Exception:
        pass

    try:
        _display_path = str(skill_dir.relative_to(_skills_dir()))
    except ValueError:
        # Skill created under skills.create_dir — not relative to the
        # profile-local root, so show the absolute path.
        _display_path = str(skill_dir)
    result = {
        "success": True,
        "message": f"Skill '{name}' created.",
        "path": _display_path,
        "skill_md": str(skill_md),
        "_change": {"description": _desc},
    }
    if category:
        result["category"] = category
    result["hint"] = (
        "To add reference files, templates, or scripts, use "
        "skill_manage(action='write_file', name='{}', file_path='references/example.md', file_content='...')".format(name)
    )
    _add_description_prompt_preview(result, content)
    _attach_lint_findings(result, skill_md)
    return result

def _edit_skill(name: str, content: str) -> Dict[str, Any]:
    """Replace the SKILL.md of any existing skill (full rewrite)."""
    err = _validate_frontmatter(content)
    if err:
        return {"success": False, "error": err}

    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    org_guard = _org_mirror_write_guard(name, existing["path"], "edit")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, existing["path"], "edit")
    if guard:
        return guard

    skill_md = existing["path"] / "SKILL.md"
    read_guard = _background_review_read_before_write_guard(
        name, skill_md, "edit", "SKILL.md"
    )
    if read_guard:
        return read_guard

    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard

    # Back up original content for rollback
    original_content = _pending_read_text(skill_md) if skill_md.exists() else None
    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard
    _pending_atomic_write_text(skill_md, content)
    try:
        published_tree_hash = _pending_assert_anchor_tree_current()
    except ValueError:
        if original_content is not None:
            _pending_assert_published_text_current(skill_md, content)
            _pending_atomic_write_text(skill_md, original_content)
        raise

    # Security scan — roll back on block
    scan_error = _security_scan_skill(existing["path"])
    if scan_error:
        if original_content is not None:
            _pending_assert_published_text_current(skill_md, content)
            _pending_atomic_write_text(skill_md, original_content)
        return {"success": False, "error": scan_error}
    try:
        _pending_assert_anchor_tree_current(published_tree_hash)
    except ValueError:
        if original_content is not None:
            _pending_assert_published_text_current(skill_md, content)
            _pending_atomic_write_text(skill_md, original_content)
        raise

    # Extract description from new content for verbose notifications
    _desc = ""
    try:
        _fm_end = re.search(r'\n---\s*\n', content[3:])
        if _fm_end:
            _parsed = yaml.safe_load(content[3:_fm_end.start() + 3])
            _desc = str(_parsed.get("description", ""))[:120]
    except Exception:
        pass

    result = {
        "success": True,
        "message": f"Skill '{name}' updated (full rewrite).",
        "path": str(existing["path"]),
        "_change": {"description": _desc},
    }
    org_note = _maybe_auto_propose_org_edit(name, existing["path"])
    if org_note:
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    _add_description_prompt_preview(result, content)
    return result

def _patch_skill(
    name: str,
    old_string: str,
    new_string: str,
    file_path: str = None,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """Targeted find-and-replace within a skill file.

    Defaults to SKILL.md. Use file_path to patch a supporting file instead.
    Requires a unique match unless replace_all is True.
    """
    if not old_string:
        # A bare "required" error is a dead end: the model cannot tell whether it
        # omitted the arg or supplied it wrongly, so it retries blindly and often
        # escapes to action='write_file', clobbering the whole skill file. Tell it
        # how to recover. Upstream: NousResearch/hermes-agent#33064.
        return {
            "success": False,
            "error": (
                "old_string is required for 'patch' and must be the EXACT text currently in the "
                "file. Read the target file first (read_file on the skill's SKILL.md, or the file "
                "named by file_path) and copy the snippet verbatim, then retry 'patch'. "
                "Do NOT fall back to action='write_file' — that rewrites the entire file and "
                "destroys unrelated content."
            ),
        }
    if new_string is None:
        return {"success": False, "error": "new_string is required for 'patch'. Use an empty string to delete matched text."}
    # No old_string == new_string guard here: fuzzy_find_and_replace already
    # rejects that with "old_string and new_string are identical"
    # (tools/fuzzy_match.py), and its error carries a file_preview this layer
    # cannot produce. Duplicating it here would only shadow the richer message.

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}

    skill_dir = existing["path"]
    org_guard = _org_mirror_write_guard(name, skill_dir, "patch")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, skill_dir, "patch")
    if guard:
        return guard

    if file_path:
        # Patching a supporting file
        err = _validate_file_path(file_path)
        if err:
            return {"success": False, "error": err}
        target, err = _resolve_skill_target(skill_dir, file_path)
        if err:
            return {"success": False, "error": err}
        assert target is not None
    else:
        # Patching SKILL.md
        target = skill_dir / "SKILL.md"

    if not target.exists():
        return {"success": False, "error": f"File not found: {target.relative_to(skill_dir)}"}

    read_guard = _background_review_read_before_write_guard(
        name,
        target,
        "patch",
        "SKILL.md" if not file_path else file_path,
    )
    if read_guard:
        return read_guard

    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard

    content = _pending_read_text(target)

    # Use the same fuzzy matching engine as the file patch tool.
    # This handles whitespace normalization, indentation differences,
    # escape sequences, and block-anchor matching — saving the agent
    # from exact-match failures on minor formatting mismatches.
    from tools.fuzzy_match import fuzzy_find_and_replace

    new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all
    )
    if match_error:
        # Show a short preview of the file so the model can self-correct
        preview = content[:500] + ("..." if len(content) > 500 else "")
        err_msg = match_error
        try:
            from tools.fuzzy_match import format_no_match_hint
            err_msg += format_no_match_hint(match_error, match_count, old_string, content)
        except Exception:
            pass
        return {
            "success": False,
            "error": err_msg,
            "file_preview": preview,
        }

    # Check size limit on the result
    target_label = "SKILL.md" if not file_path else file_path
    err = _validate_content_size(new_content, label=target_label)
    if err:
        return {"success": False, "error": err}

    # If patching SKILL.md, validate frontmatter is still intact
    if not file_path:
        err = _validate_frontmatter(new_content)
        if err:
            return {
                "success": False,
                "error": f"Patch would break SKILL.md structure: {err}",
            }

    original_content = content  # for rollback
    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard
    _pending_atomic_write_text(target, new_content)
    try:
        published_tree_hash = _pending_assert_anchor_tree_current()
    except ValueError:
        _pending_assert_published_text_current(target, new_content)
        _pending_atomic_write_text(target, original_content)
        raise

    # Security scan — roll back on block
    scan_error = _security_scan_skill(skill_dir)
    if scan_error:
        _pending_assert_published_text_current(target, new_content)
        _pending_atomic_write_text(target, original_content)
        return {"success": False, "error": scan_error}
    try:
        _pending_assert_anchor_tree_current(published_tree_hash)
    except ValueError:
        _pending_assert_published_text_current(target, new_content)
        _pending_atomic_write_text(target, original_content)
        raise

    result = {
        "success": True,
        "message": f"Patched {'SKILL.md' if not file_path else file_path} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
    }
    # Include change previews for verbose notifications
    result["_change"] = {
        "old": old_string[:200] + ("…" if len(old_string) > 200 else ""),
        "new": new_string[:200] + ("…" if len(new_string) > 200 else ""),
    }
    org_note = _maybe_auto_propose_org_edit(name, skill_dir)
    if org_note:
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    return result

def _delete_skill(name: str, absorbed_into: Optional[str] = None) -> Dict[str, Any]:
    """Delete a skill.

    ``absorbed_into`` declares intent:
      - ``None`` / missing  → caller didn't declare (legacy / non-curator path);
        accepted for backward compat but logs a warning because the curator
        classification pipeline can't tell consolidation from pruning without it.
      - ``""`` (empty)      → explicit "truly pruned, no forwarding target".
      - ``"<skill-name>"``  → content was absorbed into that umbrella; the
        target must exist on disk. Validated here so the model can't claim an
        umbrella that doesn't exist.
    """
    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    if _pending_target_anchor.get() is not None:
        return {
            "success": False,
            "error": (
                "Approved skill deletes cannot be replayed with a portable "
                "inode-bound unlink; the target was left unchanged. "
                "Reject this record or perform a separate explicit delete."
            ),
            "_fail_closed": True,
        }
    org_guard = _org_mirror_write_guard(name, existing["path"], "delete")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, existing["path"], "delete")
    if guard:
        return guard

    # Fail closed on unverified deletes during the curator consolidation pass.
    # A bare prune (no absorbed_into) from the LLM umbrella pass is the
    # fail-open behavior reported in #29912 — refuse it; keep the skill active.
    fail_closed = _curator_consolidation_delete_guard(name, absorbed_into)
    if fail_closed:
        return fail_closed

    pinned_err = _pinned_guard(name)
    if pinned_err:
        return {"success": False, "error": pinned_err}

    # Validate absorbed_into target when declared non-empty
    absorbed_target = (
        absorbed_into.strip()
        if absorbed_into is not None and isinstance(absorbed_into, str)
        else ""
    )
    is_consolidation = bool(absorbed_target)
    if is_consolidation:
        target_name = absorbed_target
        if target_name == name:
            return {
                "success": False,
                "error": f"absorbed_into='{target_name}' cannot equal the skill being deleted.",
            }
        target = _find_skill(target_name)
        if not target:
            return {
                "success": False,
                "error": (
                    f"absorbed_into='{target_name}' does not exist. "
                    f"Create or patch the umbrella skill first, then retry the delete."
                ),
            }

    skill_dir = existing["path"]
    skills_root = _containing_skills_root(skill_dir)

    # Defense-in-depth before the recursive delete (port of Kilo Code #11240).
    unsafe = _validate_delete_target(skill_dir)
    if unsafe:
        return {"success": False, "error": unsafe}

    # During the curator consolidation pass, a verified consolidation must be
    # RECOVERABLE: archival into ~/.hermes/skills/.archive/ is documented as
    # the maximum destructive action the curator may take, and
    # `hermes curator restore` promises the skill can be brought back. Route
    # through the recoverable archive primitive instead of permanent rmtree so
    # a misjudged consolidation can be undone (#29912). Foreground,
    # user-directed deletes keep their existing hard-delete semantics.
    try:
        from tools.skill_provenance import is_background_review
        curator_pass = is_background_review()
    except Exception:
        curator_pass = False

    pending_replay = _pending_target_anchor.get() is not None
    if curator_pass and not pending_replay:
        pre_image_guard = _pending_pre_image_guard(name)
        if pre_image_guard:
            return pre_image_guard
        try:
            from tools.skill_usage import archive_skill
            ok, archive_msg = archive_skill(name)
        except Exception as e:
            return {"success": False, "error": f"failed to archive '{name}': {e}"}
        if not ok:
            return {"success": False, "error": archive_msg}
        message = f"Skill '{name}' archived ({archive_msg})."
        if is_consolidation:
            message += f" Content absorbed into '{absorbed_target}'."
        return {"success": True, "message": message, "_archived": True}

    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard
    _pending_rmtree(skill_dir)

    # Clean up empty category directories only on the ordinary pathname path.
    if _pending_target_anchor.get() is None:
        parent = skill_dir.parent
        if parent != skills_root and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    if curator_pass:
        message = f"Skill '{name}' archived in the approved-write quarantine."
        if is_consolidation:
            message += f" Content absorbed into '{absorbed_target}'."
        return {"success": True, "message": message, "_archived": True}

    message = f"Skill '{name}' deleted."
    if is_consolidation:
        message += f" Content absorbed into '{absorbed_target}'."

    return {
        "success": True,
        "message": message,
    }

def _rmdir_if_empty(parent: Path, stop: Path) -> None:
    if parent != stop and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _write_file(name: str, file_path: str, file_content: str) -> Dict[str, Any]:
    """Add or overwrite a supporting file within any skill directory."""
    err = _validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}

    if not file_content and file_content != "":
        return {"success": False, "error": "file_content is required."}

    # Check size limits
    content_bytes = len(file_content.encode("utf-8"))
    if content_bytes > MAX_SKILL_FILE_BYTES:
        return {
            "success": False,
            "error": (
                f"File content is {content_bytes:,} bytes "
                f"(limit: {MAX_SKILL_FILE_BYTES:,} bytes / 1 MiB). "
                f"Consider splitting into smaller files."
            ),
        }
    err = _validate_content_size(file_content, label=file_path)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name, " Create it first with action='create'.")}
    org_guard = _org_mirror_write_guard(name, existing["path"], "write_file")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, existing["path"], "write_file")
    if guard:
        return guard

    target, err = _resolve_skill_target(existing["path"], file_path)
    if err:
        return {"success": False, "error": err}
    assert target is not None
    if target.exists():
        read_guard = _background_review_read_before_write_guard(
            name, target, "write_file", file_path
        )
        if read_guard:
            return read_guard
    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard
    if _pending_target_anchor.get() is None:
        target.parent.mkdir(parents=True, exist_ok=True)
    # Back up for rollback
    original_content = _pending_read_text(target) if target.exists() else None
    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard
    _pending_atomic_write_text(
        target,
        file_content,
        create_parents=True,
        expect_absent=original_content is None,
    )

    def _rollback_published_file() -> None:
        _pending_assert_published_text_current(target, file_content)
        if original_content is not None:
            _pending_atomic_write_text(target, original_content)
        else:
            try:
                _pending_unlink(target)
            except FileNotFoundError:
                pass

    try:
        published_tree_hash = _pending_assert_anchor_tree_current()
    except ValueError:
        _rollback_published_file()
        raise

    # Security scan — roll back on block
    scan_error = _security_scan_skill(existing["path"])
    if scan_error:
        _rollback_published_file()
        return {"success": False, "error": scan_error}
    try:
        _pending_assert_anchor_tree_current(published_tree_hash)
    except ValueError:
        _rollback_published_file()
        raise

    result = {
        "success": True,
        "message": f"File '{file_path}' written to skill '{name}'.",
        "path": str(target),
    }
    org_note = _maybe_auto_propose_org_edit(name, existing["path"])
    if org_note:
        result["org_sharing"] = org_note
        result["message"] = f"{result['message']} {org_note}"
    return result

def _remove_file(name: str, file_path: str) -> Dict[str, Any]:
    """Remove a supporting file from any skill directory."""
    err = _validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}

    skill_dir = existing["path"]
    guard = _background_review_write_guard(name, skill_dir, "remove_file")
    if guard:
        return guard

    target, err = _resolve_skill_target(skill_dir, file_path)
    if err:
        return {"success": False, "error": err}
    assert target is not None
    if not target.exists():
        # List what's actually there for the model to see
        available = []
        for subdir in ALLOWED_SUBDIRS:
            d = skill_dir / subdir
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        available.append(str(f.relative_to(skill_dir)))
        return {
            "success": False,
            "error": f"File '{file_path}' not found in skill '{name}'.",
            "available_files": available if available else None,
        }

    read_guard = _background_review_read_before_write_guard(
        name, target, "remove_file", file_path
    )
    if read_guard:
        return read_guard

    pre_image_guard = _pending_pre_image_guard(name)
    if pre_image_guard:
        return pre_image_guard

    _pending_unlink(target)

    # Clean up empty subdirectories only on the ordinary pathname-based path.
    # Approved replay keeps descriptor identity binding and leaves empty parents.
    if _pending_target_anchor.get() is None:
        parent = target.parent
        if parent != skill_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    return {
        "success": True,
        "message": f"File '{file_path}' removed from skill '{name}'.",
    }

# --- Main entry point ---------------------------------------------------------

# Set while replaying an approved staged skill write so skill_manage() does not re-gate it.
_skill_gate_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False)


def _run_write_gate(build_staging):
    """Shared write gate: None to proceed, else a JSON tool result (blocked/staged).
    ``build_staging(wa) -> (payload, gist)`` runs only when staging. Fails open if
    write_approval cannot be imported."""
    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open
    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)
    payload, gist = build_staging(wa)
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    return json.dumps({"success": True, "staged": True, "pending_id": record["id"],
                       "gist": gist, "message": decision.message}, ensure_ascii=False)


def _apply_skill_write_gate(
    action,
    name,
    *,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    background_review_read_verified: bool = False,
    **payload_kwargs,
):
    """Evaluate the skill write gate. Returns a JSON tool-result string when the
    write should NOT proceed (blocked or staged), or None to perform the real
    write. Bypassed during approved-pending replay.
    """
    if action not in {"create", "edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    if _skill_gate_bypass.get():
        return None

    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open

    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)
    if action == "delete":
        return json.dumps(
            {
                "success": False,
                "error": (
                    "Skill delete cannot be safely replayed after approval; "
                    "no pending record was created. Disable the approval gate "
                    "only for a separate explicit delete."
                ),
                "_fail_closed": True,
            },
            ensure_ascii=False,
        )

    # stage — record the full skill_manage kwargs so approval can replay it.
    payload = {"action": action, "name": name}
    payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
    gist = wa.skill_gist(
        action, name,
        content=payload_kwargs.get("content") or "",
        file_path=payload_kwargs.get("file_path") or "",
        old_string=payload_kwargs.get("old_string") or "",
        new_string=payload_kwargs.get("new_string") or "",
    )
    try:
        target_hash = _target_tree_pre_image_hash(
            name, payload_kwargs.get("category")
        )
        record = wa.stage_write(
            wa.SKILLS,
            payload,
            summary=gist,
            origin=wa.current_origin(),
            session_context=wa.collect_session_context(
                session_id=session_id,
                tool_call_id=tool_call_id,
            ),
            target_tree_pre_image_hash=target_hash,
            background_review_read_verified=(
                background_review_read_verified
                if wa.current_origin() == "background_review"
                else None
            ),
        )
    except (OSError, ValueError, wa.PendingStoreError) as exc:
        return tool_error(f"Skill write was not staged safely: {exc}", success=False)
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "gist": gist, "message": decision.message},
        ensure_ascii=False,
    )

_FLAT_OP_KEYS = ("content", "category", "file_path", "file_content", "old_string", "new_string",
                 "absorbed_into", "operations")


def _skill_manage_from(payload: Dict[str, Any], **extra) -> str:
    """Call ``skill_manage`` with the flat-shape fields (and absorbed_into/operations) of ``payload``."""
    return skill_manage(
        action=payload.get("action", ""), name=payload.get("name", ""),
        replace_all=payload.get("replace_all", False),
        **{k: payload.get(k) for k in _FLAT_OP_KEYS}, **extra)


def apply_skill_pending(
    payload: Dict[str, Any],
    *,
    expected_target_tree_pre_image_hash: Optional[str] = None,
    origin: str = "foreground",
    background_review_read_verified: bool = False,
) -> str:
    """Replay a staged skill write after revalidating bound state and origin."""
    operations = payload.get("operations")
    if operations is not None and (
        not isinstance(operations, list) or len(operations) != 1
    ):
        return json.dumps(
            {
                "success": False,
                "error": (
                    "Multi-operation skill batches cannot be replayed with "
                    "the descriptor-bound approval path."
                ),
                "_fail_closed": True,
            },
            ensure_ascii=False,
        )
    if not expected_target_tree_pre_image_hash:
        return tool_error("Pending skill write has no target pre-image hash.", success=False)
    from tools.skill_provenance import (
        reset_current_write_origin,
        set_current_write_origin,
    )

    token = _skill_gate_bypass.set(True)
    pre_image_token = _pending_apply_pre_image_hash.set(
        expected_target_tree_pre_image_hash
    )
    origin_token = set_current_write_origin(origin)
    read_guard_token = _pending_apply_read_guard_bypass.set(
        origin == "background_review" and background_review_read_verified is True
    )
    try:
        return skill_manage(
            action=payload.get("action", ""),
            name=payload.get("name", ""),
            content=payload.get("content"),
            category=payload.get("category"),
            file_path=payload.get("file_path"),
            file_content=payload.get("file_content"),
            old_string=payload.get("old_string"),
            new_string=payload.get("new_string"),
            replace_all=payload.get("replace_all", False),
            absorbed_into=payload.get("absorbed_into"),
            operations=payload.get("operations"),
        )
    finally:
        _pending_apply_read_guard_bypass.reset(read_guard_token)
        reset_current_write_origin(origin_token)
        _pending_apply_pre_image_hash.reset(pre_image_token)
        _skill_gate_bypass.reset(token)

# Sync push debounce: a burst of skill_manage writes collapses into one push on a daemon timer.
_sync_push_timer = None
_sync_push_lock = threading.Lock()
_SYNC_PUSH_DEBOUNCE_S = 5.0


def _maybe_debounced_sync_push(skill_name: str) -> None:
    """Debounced best-effort sync push after a skill write; never blocks the caller. Skills not
    opted into sync do nothing (no auth/network); ``maybe_push_skills`` enforces the access gate."""
    global _sync_push_timer
    try:
        from tools.skill_usage import is_sync_enabled
        if not is_sync_enabled(skill_name):
            return
    except Exception:
        return
    def _fire():
        with suppress(Exception):
            from tools.skills_sync_client import maybe_push_skills
            maybe_push_skills(message=f"sync: {skill_name}")
    with _sync_push_lock:
        if _sync_push_timer is not None:
            _sync_push_timer.cancel()  # only sets an Event; never raises
        _sync_push_timer = threading.Timer(_SYNC_PUSH_DEBOUNCE_S, _fire)
        _sync_push_timer.daemon = True
        _sync_push_timer.start()


def _act_patch(a):
    """Two shapes: old_string/new_string = targeted replacement (validated in _patch_skill so the
    tool and the helper give the same guidance); content alone = full rewrite (the old 'edit')."""
    if a["content"] and (a["old_string"] or a["new_string"] is not None):
        return tool_error("Pass EITHER content (full SKILL.md rewrite) OR "
                          "old_string/new_string (targeted replacement), not both.", success=False)
    if a["content"]:
        return _edit_skill(a["name"], a["content"])
    return _patch_skill(a["name"], a["old_string"], a["new_string"], a["file_path"], a["replace_all"])


# action -> handler(args dict) returning a result dict, or a tool_error JSON string for
# argument-shape errors. "edit" is a legacy alias for a full rewrite (not in the schema).
_ACTION_HANDLERS = {
    "create": lambda a: _create_skill(a["name"], a["content"], a["category"]),
    "edit": lambda a: _edit_skill(a["name"], a["content"]),
    "patch": _act_patch,
    "delete": lambda a: _delete_skill(a["name"], absorbed_into=a["absorbed_into"]),
    "write_file": lambda a: _write_file(a["name"], a["file_path"], a["file_content"]),
    "remove_file": lambda a: _remove_file(a["name"], a["file_path"])}
# action -> (arg, is_missing, error) argument-shape checks run before the handler.
_MISSING, _IS_NONE = (lambda v: not v), (lambda v: v is None)
_REQUIRED_ARGS = {
    "create": [("content", _MISSING,
                "content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).")],
    "edit": [("content", _MISSING,
              "content is required for a full rewrite. Provide the full updated SKILL.md text.")],
    "write_file": [
        ("file_path", _MISSING, "file_path is required for 'write_file'. Example: 'references/api-guide.md'"),
        ("file_content", _IS_NONE, "file_content is required for 'write_file'.")],
    "remove_file": [("file_path", _MISSING, "file_path is required for 'remove_file'.")]}


def _record_success(action, name, result, *, file_path, absorbed_into, task_id,
                    session_id, ledger_before) -> None:
    """Best-effort post-mutation side effects (never break the tool): ledger, prompt-cache
    clear, curator telemetry, debounced sync push."""
    with suppress(Exception):
        from tools import skill_ledger as _ledger
        _post = _find_skill(name)
        # delete: consolidation vs prune, and whether the recoverable archive handled it
        _evidence = ({"absorbed_into": absorbed_into, "archived": bool(result.get("_archived"))}
                     if action == "delete" else {})
        _evidence.update({k: v for k, v in (("session_id", session_id), ("file_path", file_path)) if v})
        _ledger.record_mutation(
            action, name, before=ledger_before if ledger_before is not None else [],
            after_root=_post["path"] if _post else None, evidence=_evidence)
    with suppress(Exception):
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    # Curator telemetry: only the background review fork marks a skill agent-created
    # (foreground creates belong to the user). A recoverable curator archive keeps its
    # record as STATE_ARCHIVED (`hermes curator status`/`restore`); only a hard delete forgets.
    with suppress(Exception):
        from tools.skill_usage import bump_patch, forget, record_created
        # During the curator consolidation pass, a verified consolidation must be RECOVERABLE: archival into
        # ~/.hermes/skills/.archive/ is documented as the maximum destructive action the curator may take,
        # and `hermes curator restore` promises the skill can be brought back. Route through the recoverable
        # archive primitive instead of permanent rmtree so a misjudged consolidation can be undone (#29912).
        # Foreground, user-directed deletes keep their existing hard-delete semantics.
        from tools.skill_provenance import is_background_review
        if action == "create":
            record_created(name, agent_created=is_background_review(),
                           task_id=task_id, session_id=session_id)
        elif action in {"patch", "edit", "write_file", "remove_file"}:
            bump_patch(name, action=action, task_id=task_id, session_id=session_id)
        elif action == "delete" and not result.get("_archived"):
            forget(name)
    # Only AFTER the write gate passed (staged writes returned early): never push un-reviewed content.
    with suppress(Exception):
        _maybe_debounced_sync_push(name)


def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    operations=None,
) -> str:
    """Run one skill operation while holding the cross-runtime mutation lock."""
    try:
        with _skill_write_lock():
            if operations is not None:
                return _skill_manage_unlocked(
                    action=action,
                    name=name,
                    content=content,
                    category=category,
                    file_path=file_path,
                    file_content=file_content,
                    old_string=old_string,
                    new_string=new_string,
                    replace_all=replace_all,
                    absorbed_into=absorbed_into,
                    task_id=task_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    operations=operations,
                )
            with _pending_target_anchor_context(name, category):
                return _skill_manage_unlocked(
                    action=action,
                    name=name,
                    content=content,
                    category=category,
                    file_path=file_path,
                    file_content=file_content,
                    old_string=old_string,
                    new_string=new_string,
                    replace_all=replace_all,
                    absorbed_into=absorbed_into,
                    task_id=task_id,
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    operations=operations,
                )
    except (OSError, ValueError) as exc:
        return tool_error(f"Skill write lock could not be secured: {exc}", success=False)

# --- OpenAI Function-Calling Schema -------------------------------------------

SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    # ONE advertised call shape (memory-tool pattern): the call IS an operations
    # array. The legacy flat shape (top-level action/name/content/...) is still
    # ACCEPTED for old transcripts and staged-write replay, but not advertised.
    "description": (
        "Create, update, or delete skills — your procedural memory for "
        "recurring task types. The call is an operations array (a single "
        "edit is a list of one); it applies atomically — any failure rolls "
        "every touched skill back. Ops: create (full SKILL.md; lands in "
        f"{_display_create_dir()}; must precede that skill's other "
        "ops), patch (targeted old_string/new_string fix — preferred; "
        "content alone REPLACES the whole file, read it via skill_view() "
        "first), write_file/remove_file (supporting files), delete (sole "
        "op only). Existing skills are modified wherever they live. Keep "
        "the description's first 57 chars a self-contained trigger: 'Use "
        "when <trigger>. <one-line behavior>.' Write lessons, not logs: "
        "imperative rule + why, no PR numbers/dates/incident narration, one "
        "rule per lesson, references/ named by topic (extend before adding). "
        "skill_view() shows format conventions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "Ordered ops; each names its target skill.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Skill name (lowercase, hyphens/underscores, "
                                "max 64 chars); an existing skill's name "
                                "unless creating."
                            )
                        },
                        "action": {
                            "type": "string",
                            "enum": ["create", "patch", "delete", "write_file", "remove_file"]
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Full SKILL.md text (YAML frontmatter + "
                                "markdown body) for create, or a full "
                                "rewrite on patch."
                            )
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category subdir for create (e.g. 'devops')."
                        },
                        # patch args: same fuzzy-matching semantics as the
                        # `patch` tool — teach only skill-specific facts here.
                        "old_string": {
                            "type": "string",
                            "description": "Text to find (patch; same matching semantics as the patch tool)."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement (patch); empty string deletes the match."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "patch: replace all occurrences (default false)."
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Path RELATIVE to the skill's own directory, "
                                "e.g. 'references/api.md' — no leading slash, "
                                "never absolute. write_file/remove_file: "
                                "required; first segment references/, "
                                "templates/, scripts/, or assets/. patch: "
                                "optional (default SKILL.md)."
                            )
                        },
                        "file_content": {
                            "type": "string",
                            "description": "Content for write_file."
                        }
                    },
                    "required": ["name", "action"]
                }
            },
            # Also accepted, never advertised: the legacy flat single-op fields, and
            # `absorbed_into` on delete ops (curator-only vocabulary; the curator's
            # prompt documents it and the delete guard's error re-teaches it).
        },
        "required": ["operations"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="skill_manage", toolset="skills", schema=SKILL_MANAGE_SCHEMA, emoji="📝",
    handler=lambda args, **kw: _skill_manage_from(
        args, task_id=kw.get("task_id"), session_id=kw.get("session_id")))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'mark_background_review_skill_read': ('tools.skill_manager_guards', 'mark_background_review_skill_read'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
