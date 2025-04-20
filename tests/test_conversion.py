"""Defines a dummy test."""

import tempfile
from pathlib import Path

import pytest

from urdf2mjcf.convert import convert_urdf_to_mjcf


@pytest.mark.slow
def test_conversion_output(tmpdir: Path) -> None:
    urdf_path = Path(__file__).parent / "sample" / "robot.urdf"
    mjcf_path = tmpdir / "robot.mjcf"

    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        copy_meshes=False,
        metadata_file=urdf_path.parent / "metadata.json",
    )

    # After making a change, put a breakpoint here and make sure you try out
    # the model in Mujoco before committing changes.
    assert mjcf_path.exists()


if __name__ == "__main__":
    # python -m tests.test_conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        test_conversion_output(Path(temp_dir))
