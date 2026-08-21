"""The firewall is the clean-room's load-bearing control, so it is tested for
the ways it could silently stop protecting: a moved archive, a symlink, a
relative path, and a missing marker file."""

import os
from pathlib import Path

import pytest

from pgeom.firewall import (
    ARCHIVE_MARKER,
    ARCHIVE_ROOT_ENV,
    ArchiveAccessError,
    archive_reason,
    guard_path,
    is_archive_path,
)


def test_rejects_path_under_configured_root(monkeypatch, tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(root))
    assert is_archive_path(root / "exp" / "runs" / "x.json")


def test_rejects_by_marker_even_when_root_is_wrong(monkeypatch, tmp_path):
    """A moved archive must still be caught. This is the failure mode that a
    single hardcoded absolute path would miss entirely."""
    moved = tmp_path / "somewhere_else"
    (moved / "exp" / "data").mkdir(parents=True)
    (moved / ARCHIVE_MARKER).write_text("archive")
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(tmp_path / "does_not_exist"))
    assert is_archive_path(moved / "exp" / "data" / "prompts.jsonl")
    assert "marker" in archive_reason(moved / "exp" / "data" / "prompts.jsonl")


def test_rejects_symlink_pointing_into_archive(monkeypatch, tmp_path):
    root = tmp_path / "archive"
    (root / "exp").mkdir(parents=True)
    target = root / "exp" / "secret.json"
    target.write_text("{}")
    link = tmp_path / "innocent.json"
    link.symlink_to(target)
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(root))
    assert is_archive_path(link)


def test_allows_clean_path(monkeypatch, tmp_path):
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(tmp_path / "archive"))
    clean = tmp_path / "study" / "prompts.jsonl"
    assert not is_archive_path(clean)
    assert guard_path(clean) == clean.resolve()


def test_guard_raises_with_actionable_message(monkeypatch, tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(root))
    with pytest.raises(ArchiveAccessError) as exc:
        guard_path(root / "x.json", role="prompt file")
    assert "prompt file" in str(exc.value)
    assert "clean-room violation" in str(exc.value)


def test_real_archive_is_rejected_by_default_configuration():
    """No monkeypatching: the shipped default must reject the real archive."""
    real = "/Users/chaoticcooties/Desktop/renegade/archive"
    if not Path(real).exists():
        pytest.skip("archive not present on this machine")
    assert is_archive_path(real)
    assert is_archive_path(Path(real) / "renegade" / "runs" / "anything.json")


def test_new_repo_is_not_rejected():
    assert not is_archive_path(Path(__file__).resolve().parents[1])


# --- Shared-volume containment ----------------------------------------------
# The study reuses a network volume that already holds a previous study's work.
# Marker ascent cannot protect there, because the volume carries no marker file,
# so containment is the only guard and it must be exercised directly.

from pgeom.firewall import (  # noqa: E402
    SHARED_ROOT_ENV,
    STUDY_SUBDIR_ENV,
    shared_root_policy,
)


def test_shared_volume_paths_outside_the_study_subdir_fail_closed(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    (root / "tara_gcg_runs").mkdir(parents=True)
    (root / "pgeom").mkdir()
    monkeypatch.setenv(SHARED_ROOT_ENV, str(root))
    monkeypatch.setenv(STUDY_SUBDIR_ENV, str(root / "pgeom"))
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(tmp_path / "no_archive_here"))

    assert is_archive_path(root / "tara_gcg_runs" / "suffixes.json")
    assert is_archive_path(root / "stray.pt")
    assert "shared storage" in archive_reason(root / "stray.pt")


def test_study_subdir_on_the_shared_volume_is_allowed(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    (root / "pgeom" / "phase3").mkdir(parents=True)
    monkeypatch.setenv(SHARED_ROOT_ENV, str(root))
    monkeypatch.setenv(STUDY_SUBDIR_ENV, str(root / "pgeom"))
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(tmp_path / "no_archive_here"))
    assert not is_archive_path(root / "pgeom" / "phase3" / "checkpoint.json")


def test_containment_is_on_by_default(monkeypatch):
    """Absent configuration, the protective default applies rather than nothing."""
    monkeypatch.delenv(SHARED_ROOT_ENV, raising=False)
    monkeypatch.delenv(STUDY_SUBDIR_ENV, raising=False)
    policy = shared_root_policy()
    assert policy is not None
    root, subdir = policy
    assert str(root) == "/workspace"
    assert str(subdir) == "/workspace/pgeom"


def test_misconfigured_subdir_outside_root_is_rejected_loudly(monkeypatch, tmp_path):
    """A subdir outside the root would whitelist everything, so it must raise."""
    monkeypatch.setenv(SHARED_ROOT_ENV, str(tmp_path / "workspace"))
    monkeypatch.setenv(STUDY_SUBDIR_ENV, str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="not inside shared root"):
        shared_root_policy()


def test_containment_can_be_disabled_only_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv(SHARED_ROOT_ENV, "none")
    monkeypatch.setenv(ARCHIVE_ROOT_ENV, str(tmp_path / "no_archive"))
    assert shared_root_policy() is None
    assert not is_archive_path("/workspace/tara_gcg_runs/x.json")
