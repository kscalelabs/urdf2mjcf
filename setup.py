# mypy: disable-error-code="import-untyped, import-not-found"
#!/usr/bin/env python
"""Setup script for the project."""

import re
from typing import List

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("urdf2mjcf/requirements.txt", "r", encoding="utf-8") as f:
    requirements: List[str] = f.read().splitlines()


with open("urdf2mjcf/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: List[str] = f.read().splitlines()


with open("urdf2mjcf/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in urdf2mjcf/__init__.py"
version: str = version_re.group(1)


setup(
    name="urdf2mjcf",
    version=version,
    description="The urdf2mjcf project",
    author="Benjamin Bolte",
    url="https://github.com/kscalelabs/urdf2mjcf",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "urdf2mjcf=urdf2mjcf.convert:main",
        ],
    },
)
