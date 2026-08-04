"""Mark the server smoke tests like the codegen suites they are.

This module is collected by a bare `pytest` from the repo root but lives
outside `tests/`, so the suite's conftest never saw it — its WASM-building
tests ran `emcc` even under `-m "not cpp"`. Everything here is
toolchain-or-build shaped, so the whole module gets both markers.
"""


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        item.add_marker(pytest.mark.cpp)
        item.add_marker(pytest.mark.slow)
