import re
import subprocess
import sys
from pathlib import Path

import sarrl


def test_package_version_matches_pyproject():
    text = Path("pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == sarrl.__version__


def test_readme_and_verification_document_ci():
    readme = Path("README.md").read_text()
    verification = Path("docs/verification.md").read_text()

    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "Python 3.10, 3.11 and 3.12" in verification


def test_documented_script_invocations_start():
    # `python tools/x.py` puts tools/ itself on sys.path, so a script that
    # imports from the `tools` package without a fallback only runs as
    # `python -m tools.x`. Only scripts that import from it are launched.
    invocation = re.compile(r"python3? tools/(\w+)\.py")
    docs = [Path("README.md"), *sorted(Path("docs").glob("*.md"))]
    names = {name for doc in docs for name in invocation.findall(doc.read_text())}
    for name in sorted(names):
        script = Path("tools", f"{name}.py")
        if not re.search(r"^\s*(from|import) tools\b", script.read_text(), re.MULTILINE):
            continue
        result = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, (name, result.stderr)
