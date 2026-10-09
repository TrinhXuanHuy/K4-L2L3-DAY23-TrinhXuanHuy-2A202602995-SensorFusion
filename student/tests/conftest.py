"""Shared student workspace setup and exercise-aware self-check handling."""

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest


_STUDENT_ROOT = Path(__file__).resolve().parents[1]
_EXERCISE_MODULES = {"kalman", "association", "camera_fusion", "track_management"}


@pytest.fixture(scope="session")
def workspace_modules():
    """Load the student workspace, respecting DAY23_STUDENT_ROOT overrides."""
    root = Path(os.environ.get("DAY23_STUDENT_ROOT", _STUDENT_ROOT)).resolve()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DAY23_STUDENT_ROOT", str(root))
        patch.setenv("FUSION_LAB_PLATFORM", str(_STUDENT_ROOT.parent / "platform"))
        from fusion_lab.workspace_loader import load_workspace

        yield load_workspace()


def pytest_configure(config):
    """Register the marker that tags Part E–H behavioural checks."""
    config.addinivalue_line(
        "markers", "student_exercise: behavioural check of a Part E–H learner function"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Treat only unfinished student E–H TODO exceptions as expected failures."""
    outcome = yield
    report = outcome.get_result()
    root = Path(os.environ.get("DAY23_STUDENT_ROOT", _STUDENT_ROOT)).resolve()
    if (
        report.when != "call"
        or root != _STUDENT_ROOT
        or call.excinfo is None
        or not call.excinfo.errisinstance(NotImplementedError)
    ):
        return
    for entry in call.excinfo.traceback:
        path = Path(str(entry.path)).resolve()
        if (
            path.parent == _STUDENT_ROOT / "workspace"
            and path.stem in _EXERCISE_MODULES
            and "TODO: implement" in str(call.excinfo.value)
        ):
            report.outcome = "skipped"
            report.wasxfail = "Part E–H student TODO is not implemented yet"
            return
