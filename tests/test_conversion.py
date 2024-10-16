"""Defines a dummy test."""

import tempfile
from pathlib import Path

import pytest

from urdf2mjcf.convert import convert_urdf_to_mjcf


@pytest.mark.slow
def test_conversion(tmpdir: Path) -> None:
    urdf_path = Path(__file__).parent / "sample" / "robot.urdf"
    mjcf_path = tmpdir / "robot.mjcf"
    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        # copy_meshes=True,
        copy_meshes=False,
    )
    assert mjcf_path.exists()


if __name__ == "__main__":
    # python -m tests.test_conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        test_conversion(Path(temp_dir))