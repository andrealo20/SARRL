import re
from pathlib import Path

import sarrl


def test_package_version_matches_pyproject():
    text = Path("pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == sarrl.__version__


def test_readme_and_verification_report_current_test_count():
    count = sum(
        1
        for path in Path("tests").glob("test_*.py")
        for line in path.read_text().splitlines()
        if line.startswith("def test_")
    )
    assert count == 71
    assert "69/69" not in Path("README.md").read_text()
