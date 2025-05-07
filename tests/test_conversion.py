"""Defines a dummy test."""

import json
import tempfile
from pathlib import Path

import pytest

from urdf2mjcf.convert import convert_urdf_to_mjcf
from urdf2mjcf.model import ActuatorMetadata, JointMetadata


@pytest.mark.slow
def test_conversion_output(tmpdir: Path) -> None:
    urdf_path = Path(__file__).parent / "sample" / "robot.urdf"
    mjcf_path = tmpdir / "robot.mjcf"

    # Load joint metadata
    joint_metadata_path = urdf_path.parent / "joint_metadata.json"
    with open(joint_metadata_path, "r") as f:
        joint_metadata = json.load(f)["joint_name_to_metadata"]
        for key, value in joint_metadata.items():
            joint_metadata[key] = JointMetadata.from_dict(value)

    # Load actuator metadata
    actuator_path = urdf_path.parent / "actuators" / "motor.json"
    with open(actuator_path, "r") as f:
        motor_data = json.load(f)
        actuator_type = motor_data["actuator_type"]
        actuator_metadata = {actuator_type: ActuatorMetadata.from_dict(motor_data)}

    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        copy_meshes=False,
        metadata_file=urdf_path.parent / "metadata.json",
        joint_metadata=joint_metadata,
        actuator_metadata=actuator_metadata,
    )

    # After making a change, put a breakpoint here and make sure you try out
    # the model in Mujoco before committing changes.
    assert mjcf_path.exists()


if __name__ == "__main__":
    # python -m tests.test_conversion
    with tempfile.TemporaryDirectory() as temp_dir:
        test_conversion_output(Path(temp_dir))
