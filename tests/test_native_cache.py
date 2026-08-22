"""Native-library cache discipline shared by every ``.so`` Manta dlopens.

Doctrine: hot kernels are cached by generated model, compiler toolchain, and
native CPU identity. These tests pin that the one helper implementing that
key is the only compile-and-publish path (CasADi externals, the filter-replay
runner, the HPIPM and OSQP bridges) and that the key moves with the toolchain.
"""

from __future__ import annotations

import re
import shutil
import sys
import sysconfig
from pathlib import Path

import casadi as ca
import pytest

from manta import EKF, TargetFilterReplay, compile_functions
from manta.codegen.numpy import _compile, _filter_replay
from manta.control import _hpipm, _osqp
from tests.test_filter_replay import _world as _replay_world

_PROBE_SOURCE = "double manta_probe(double x) { return 2.0 * x; }\n"


def _require_cc():
    if shutil.which("cc") is None:
        pytest.skip("no C compiler on PATH")


def test_native_library_key_moves_with_the_toolchain_identity(
    monkeypatch, tmp_path,
):
    _require_cc()
    monkeypatch.setattr(_compile, "_cache_dir", lambda: str(tmp_path))
    identities = iter([b"toolchain-A", b"toolchain-A", b"toolchain-B"])
    monkeypatch.setattr(
        _compile, "toolchain_identity",
        lambda compiler, flags, *, timeout_s: next(identities))
    build = {"stem": "probe", "what": "probe", "compiler_flags": ("-O1",),
             "timeout_s": 30.0}
    first = _compile.build_native_library(_PROBE_SOURCE, **build)
    again = _compile.build_native_library(_PROBE_SOURCE, **build)
    other = _compile.build_native_library(_PROBE_SOURCE, **build)
    assert first.rebuilt and not again.rebuilt
    assert first.path == again.path and first.key == again.key
    assert other.key != first.key and other.path != first.path
    assert other.rebuilt
    assert Path(first.path).exists() and Path(other.path).exists()
    # Every published entry sits in the shared layout: <cache>/<stem>_<key>.so
    assert Path(first.path).name == f"probe_{first.key}.so"
    assert Path(first.path).parent == tmp_path


def test_toolchain_identity_covers_compiler_abi_and_native_cpu():
    _require_cc()
    compiler = shutil.which("cc")
    generic = _compile.toolchain_identity(compiler, ("-O3",), timeout_s=10.0)
    native = _compile.toolchain_identity(
        compiler, ("-O3", "-march=native"), timeout_s=10.0)
    for expected in (
        Path(compiler).resolve().as_posix(), sys.version, ca.__version__,
        str(sysconfig.get_config_var("SOABI")),
    ):
        assert expected.encode() in generic
    assert _compile.cpu_identity().encode() in native
    assert _compile.cpu_identity().encode() not in generic
    assert native != generic


def test_build_flags_and_link_line_are_part_of_the_key(monkeypatch):
    monkeypatch.setattr(
        _compile, "toolchain_identity",
        lambda compiler, flags, *, timeout_s: b"fixed")
    base = {"stem": "k", "link_args": (), "identity_salt": b"",
            "toolchain": b"fixed"}
    key = _compile.native_library_key(
        _PROBE_SOURCE, compiler_flags=("-O1",), **base)
    assert key != _compile.native_library_key(
        _PROBE_SOURCE, compiler_flags=("-O3",), **base)
    assert key != _compile.native_library_key(
        _PROBE_SOURCE, **{**base, "link_args": ("-lm",)},
        compiler_flags=("-O1",))
    assert key != _compile.native_library_key(
        _PROBE_SOURCE, **{**base, "identity_salt": b"header"},
        compiler_flags=("-O1",))
    assert key != _compile.native_library_key(
        _PROBE_SOURCE, **{**base, "toolchain": b"other"},
        compiler_flags=("-O1",))


def test_missing_compiler_and_unwritable_cache_fail_loudly(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(_compile.shutil, "which", lambda _name: None)
    with pytest.raises(_compile.CompilationError, match="no 'cc' compiler"):
        _compile.build_native_library(
            _PROBE_SOURCE, stem="probe", what="probe",
            compiler_flags=("-O1",), timeout_s=30.0)
    blocked = tmp_path / "occupied"
    blocked.write_text("not a directory")
    monkeypatch.setattr(_compile, "_cache_dir", lambda: str(blocked))
    monkeypatch.setattr(
        _compile, "toolchain_identity",
        lambda compiler, flags, *, timeout_s: b"fixed")
    with pytest.raises(_compile.CompilationError, match="cannot create.*cache"):
        _compile.build_native_library(
            _PROBE_SOURCE, stem="probe", what="probe",
            compiler_flags=("-O1",), timeout_s=30.0, compiler="/usr/bin/cc")


def test_compiler_errors_name_the_library_and_keep_the_cache_clean(
    monkeypatch, tmp_path,
):
    _require_cc()
    monkeypatch.setattr(_compile, "_cache_dir", lambda: str(tmp_path))
    with pytest.raises(_compile.CompilationError,
                       match="probe compilation failed with exit code"):
        _compile.build_native_library(
            "this is not C\n", stem="probe", what="probe",
            compiler_flags=("-O1",), timeout_s=30.0)
    assert list(tmp_path.iterdir()) == []


_NATIVE_SITES = {
    "manta/codegen/numpy/_compile.py",
    "manta/codegen/numpy/_filter_replay.py",
    "manta/control/_hpipm.py",
    "manta/control/_osqp.py",
}


def test_only_the_shared_helper_invokes_the_compiler():
    """No cache site may grow its own compiler call or cache-key scheme."""
    root = Path(__file__).resolve().parent.parent
    for relative in sorted(_NATIVE_SITES - {"manta/codegen/numpy/_compile.py"}):
        text = (root / relative).read_text(encoding="utf-8")
        assert "build_native_library(" in text, relative
        assert "subprocess.run(" not in text, relative
        assert not re.search(r"hashlib\.sha\d+\(", text), relative
        assert "_cache_dir" not in text, relative
    compile_text = (root / "manta/codegen/numpy/_compile.py").read_text(
        encoding="utf-8")
    assert compile_text.count("-fPIC") == 1


def test_every_native_cache_builds_through_the_shared_helper(monkeypatch):
    _require_cc()
    stems: list[str] = []
    real = _compile.build_native_library

    def recording(source, *, stem, **options):
        stems.append(stem)
        return real(source, stem=stem, **options)

    for module in (_compile, _filter_replay, _hpipm, _osqp):
        monkeypatch.setattr(module, "build_native_library", recording)
    monkeypatch.setattr(_compile, "_EXTERNAL_CACHE", {})
    monkeypatch.setattr(_filter_replay, "_NATIVE_CACHE", {})
    monkeypatch.setattr(_hpipm, "_LIBRARY", None)
    monkeypatch.setattr(_osqp, "_LIBRARY", None)

    x = ca.MX.sym("native_cache_x")
    compile_functions({"probe": ca.Function("native_cache_probe", [x], [x + 1])},
                      optimization="startup")
    TargetFilterReplay(EKF(_replay_world(), gates=9.0),
                       max_operations=2, max_checkpoints=1)
    _hpipm._library()
    _osqp._library()
    assert stems == ["mod", "replay", "manta_hpipm", "manta_osqp"]
