"""Clean-room firewall.

Scientific programs fail closed when an input resolves inside the archived
Renegade experiments. Two independent detectors run on every check, because a
single hardcoded absolute path silently stops protecting anything the moment a
directory is moved:

1. containment of the fully resolved path under a configured archive root;
2. marker ascent, which walks the resolved path upward looking for the archive
   marker file.

Detector 2 keeps working after a rename or move; detector 1 keeps working if the
marker file is ever removed. A path is rejected if *either* fires.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Name of the file that marks an archive root. Presence of this file in any
#: ancestor directory of a candidate path means the path is archive history.
ARCHIVE_MARKER = "ARCHIVE_MANIFEST.md"

#: Environment variable holding os.pathsep-separated archive roots. Set this
#: rather than editing the default when the archive moves.
ARCHIVE_ROOT_ENV = "PGEOM_ARCHIVE_ROOTS"

DEFAULT_ARCHIVE_ROOTS = ("/Users/chaoticcooties/Desktop/renegade/archive",)

#: Ascent is bounded so a pathological path cannot spin. No real repository
#: nests anywhere near this deep.
MAX_ASCENT = 64

#: A shared storage root that also holds work from another study, and the one
#: subdirectory inside it this study may touch. Everything else under the root
#: fails closed.
#:
#: This exists because the study reuses the pre-existing `tara_gcg` network
#: volume rather than a fresh one. Marker ascent cannot help there: the volume
#: carries no ARCHIVE_MANIFEST.md, so without this rule a stray path into the
#: prior work would look perfectly clean.
SHARED_ROOT_ENV = "PGEOM_SHARED_ROOT"
STUDY_SUBDIR_ENV = "PGEOM_STUDY_SUBDIR"
DEFAULT_SHARED_ROOT = "/workspace"
DEFAULT_STUDY_SUBDIR = "/workspace/pgeom"


class ArchiveAccessError(RuntimeError):
    """Raised when a scientific input resolves inside the archive."""


def configured_archive_roots() -> tuple[Path, ...]:
    """Return the configured archive roots, resolved, existing ones first.

    Roots that do not exist are still returned: a missing root must not quietly
    disable detector 1, because "the archive moved" is exactly the case where
    the firewall matters most.
    """
    raw = os.environ.get(ARCHIVE_ROOT_ENV)
    parts = raw.split(os.pathsep) if raw else list(DEFAULT_ARCHIVE_ROOTS)
    roots: list[Path] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        roots.append(Path(part).expanduser().resolve(strict=False))
    return tuple(roots)


def _under_root(resolved: Path, root: Path) -> bool:
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    return True


def shared_root_policy() -> tuple[Path, Path] | None:
    """Return (shared_root, study_subdir), or None when no shared root is set.

    Returns None only when the shared root is explicitly disabled, so the
    protective default applies unless someone turns it off deliberately.
    """
    root = os.environ.get(SHARED_ROOT_ENV, DEFAULT_SHARED_ROOT)
    if not root or root.lower() in {"none", "off", "disabled"}:
        return None
    subdir = os.environ.get(STUDY_SUBDIR_ENV, DEFAULT_STUDY_SUBDIR)
    root_p = Path(root).expanduser().resolve(strict=False)
    sub_p = Path(subdir).expanduser().resolve(strict=False)
    if not _under_root(sub_p, root_p):
        raise ValueError(
            f"study subdirectory {sub_p} is not inside shared root {root_p}; "
            "the containment rule would let every path through"
        )
    return root_p, sub_p


def _marker_ancestor(resolved: Path) -> Path | None:
    """Return the nearest ancestor containing the archive marker, if any."""
    current = resolved if resolved.is_dir() else resolved.parent
    for _ in range(MAX_ASCENT):
        if (current / ARCHIVE_MARKER).is_file():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


def archive_reason(path: str | os.PathLike[str]) -> str | None:
    """Return why ``path`` is archive history, or None if it is clean.

    The path does not need to exist. Symlinks are resolved first, so a symlink
    planted outside the archive that points into it is still rejected.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    for root in configured_archive_roots():
        if _under_root(resolved, root):
            return f"path resolves under configured archive root {root}"
    marker = _marker_ancestor(resolved)
    if marker is not None:
        return f"path has archive marker {ARCHIVE_MARKER} at ancestor {marker}"

    policy = shared_root_policy()
    if policy is not None:
        root, subdir = policy
        if _under_root(resolved, root) and not _under_root(resolved, subdir):
            return (
                f"path is on shared storage {root} but outside this study's "
                f"subdirectory {subdir}; that space holds another study's work"
            )
    return None


def is_archive_path(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` resolves inside the archive."""
    return archive_reason(path) is not None


def guard_path(path: str | os.PathLike[str], *, role: str = "input") -> Path:
    """Return the resolved path, or raise :class:`ArchiveAccessError`.

    Call this on every path a scientific program reads or writes. ``role`` is
    quoted in the error so a failure names the offending study input.
    """
    reason = archive_reason(path)
    if reason is not None:
        raise ArchiveAccessError(
            f"clean-room violation: refusing to use {role} {path!r}: {reason}. "
            "Archived experiments are read-only scientific history, not study data. "
            "The only permitted archive access is the standalone corpus-overlap audit."
        )
    return Path(path).expanduser().resolve(strict=False)


def guard_paths(paths, *, role: str = "input") -> list[Path]:
    """Guard an iterable of paths, preserving order."""
    return [guard_path(p, role=role) for p in paths]
