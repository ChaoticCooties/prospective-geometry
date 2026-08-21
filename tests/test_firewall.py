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
