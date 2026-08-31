import re
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
