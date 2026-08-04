from __future__ import annotations

import errno
import os
import sys

import pytest

from forge import file_evidence, krea_runtime


def test_descriptor_reader_rejects_final_component_symlink(tmp_path):
    target = tmp_path / "record.json"
    target.write_bytes(b"{}")
    link = tmp_path / "record-link.json"
    link.symlink_to(target)

    with pytest.raises(file_evidence.RegularFileError):
        file_evidence.read_regular_bytes(
            str(link),
            label="record",
            maximum_size=1024,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="openat2 is Linux-only"
)
def test_linux_descriptor_reader_rejects_symlink_in_parent_path(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "record.json").write_bytes(b"{}")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(file_evidence.RegularFileError):
        file_evidence.read_regular_bytes(
            str(linked_parent / "record.json"),
            label="record",
            maximum_size=1024,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="openat2 is Linux-only"
)
def test_krea_attestation_reader_reuses_full_path_guard(tmp_path):
    real_parent = tmp_path / "real-runtime"
    real_parent.mkdir()
    (real_parent / "identity.json").write_bytes(b"{}")
    linked_parent = tmp_path / "linked-runtime"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(krea_runtime.KreaRuntimeContractError):
        krea_runtime._read_regular_attestation(
            str(linked_parent / "identity.json"),
            "runtime identity",
        )


def test_openat2_falls_back_only_when_kernel_api_is_unavailable(
    tmp_path, monkeypatch
):
    target = tmp_path / "record.json"
    target.write_bytes(b"{}")
    host_uname = os.uname()
    monkeypatch.setattr(file_evidence.sys, "platform", "linux")
    monkeypatch.setattr(file_evidence.os, "uname", lambda: host_uname)

    def unavailable(_path, _flags):
        raise OSError(errno.ENOSYS, "not implemented")

    monkeypatch.setattr(file_evidence, "_openat2_no_symlinks", unavailable)

    assert file_evidence.read_regular_bytes(
        str(target),
        label="record",
        maximum_size=1024,
    ) == b"{}"


def test_openat2_path_policy_error_does_not_fall_back(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b"{}")
    host_uname = os.uname()
    monkeypatch.setattr(file_evidence.sys, "platform", "linux")
    monkeypatch.setattr(file_evidence.os, "uname", lambda: host_uname)

    def policy_failure(_path, _flags):
        raise OSError(errno.ELOOP, "symlink rejected")

    monkeypatch.setattr(file_evidence, "_openat2_no_symlinks", policy_failure)

    with pytest.raises(file_evidence.RegularFileError):
        file_evidence.read_regular_bytes(
            str(target),
            label="record",
            maximum_size=1024,
        )
