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
    )
    assert mjcf_path.exists()


@pytest.mark.slow
def test_conversion_no_frc_limit(tmpdir: Path) -> None:
    urdf_path = Path(__file__).parent / "sample" / "robot.urdf"
    mjcf_path = tmpdir / "robot.mjcf"

    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        no_frc_limit=True,
        copy_meshes=False,
        default_position=[0.0] * 17,
    )

    # Compare the outputted MJCF with the expected XML
    expected_mjcf_path = Path(__file__).parent / "sample" / "robot_test.xml"
    with open(mjcf_path, "r") as output_file, open(expected_mjcf_path, "r") as expected_file:
        output_content = output_file.read()
        expected_content = expected_file.read()

        assert output_content == expected_content, "The output MJCF does not match the expected XML."


if __name__ == "__main__":
    # python -m tests.test_conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        test_conversion_output(Path(temp_dir))
        test_conversion_no_frc_limit(Path(temp_dir))
