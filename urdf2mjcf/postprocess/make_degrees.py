"""Defines a post-processing function that converts angles from radians to degrees."""

import argparse
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def convert_radians_to_degrees(value: str) -> str:
    """Convert a space-separated string of numbers from radians to degrees.

    Args:
        value: Space-separated string of numbers in radians

    Returns:
        Space-separated string of numbers in degrees
    """
    try:
        nums = [float(x) for x in value.split()]
        degrees = [math.degrees(x) for x in nums]
        return " ".join(f"{x:.6f}" for x in degrees)
    except (ValueError, TypeError):
        return value


def update_compiler_angle(root: ET.Element) -> None:
    """Update the compiler angle attribute from radian to degree.

    Args:
        root: The root element of the MJCF file.
    """
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("angle", "degree")
        logger.info("Updated compiler angle from radian to degree")


def update_joint_limits(root: ET.Element) -> None:
    """Update joint limits from radians to degrees.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all joint elements with range attribute
    joints = root.findall(".//joint[@range]")
    for joint in joints:
        range_str = joint.get("range")
        if range_str:
            joint.set("range", convert_radians_to_degrees(range_str))
            logger.info("Updated joint %s range from radians to degrees", joint.get("name", "unknown"))


def update_default_joint_limits(root: ET.Element) -> None:
    """Update default joint limits from radians to degrees.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all default joint elements with range attribute
    default_joints = root.findall(".//default/joint[@range]")
    for joint in default_joints:
        range_str = joint.get("range")
        if range_str:
            joint.set("range", convert_radians_to_degrees(range_str))
            logger.info("Updated default joint range from radians to degrees")


def update_default_motor_limits(root: ET.Element) -> None:
    """Update default motor control ranges from radians to degrees.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all default motor elements with ctrlrange attribute
    default_motors = root.findall(".//default/motor[@ctrlrange]")
    for motor in default_motors:
        ctrlrange_str = motor.get("ctrlrange")
        if ctrlrange_str:
            motor.set("ctrlrange", convert_radians_to_degrees(ctrlrange_str))
            logger.info("Updated default motor ctrlrange from radians to degrees")


def update_rpy_attributes(root: ET.Element) -> None:
    """Update rpy attributes from radians to degrees.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all elements with rpy attribute
    elements_with_rpy = root.findall(".//*[@rpy]")
    for element in elements_with_rpy:
        rpy_str = element.get("rpy")
        if rpy_str:
            element.set("rpy", convert_radians_to_degrees(rpy_str))
            logger.info("Updated rpy attribute from radians to degrees")


def update_joint_axes(root: ET.Element) -> None:
    """Update joint axes from radians to degrees.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all joint elements with axis attribute
    joints = root.findall(".//joint[@axis]")
    for joint in joints:
        axis_str = joint.get("axis")
        if axis_str:
            # The axis attribute is a direction vector, not an angle
            # We don't need to convert it to degrees
            pass


def make_degrees(mjcf_path: str | Path) -> None:
    """Convert angles from radians to degrees in the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    update_compiler_angle(root)
    update_joint_limits(root)
    update_default_joint_limits(root)
    update_default_motor_limits(root)
    update_rpy_attributes(root)
    update_joint_axes(root)
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    make_degrees(args.mjcf_path)


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.make_degrees
    main()
