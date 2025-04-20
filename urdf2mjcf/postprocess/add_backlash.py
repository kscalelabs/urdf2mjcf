"""Defines a post-processing function that adds backlash joints to hinge joints."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def add_backlash_default(root: ET.Element, backlash: float, backlash_damping: float) -> None:
    """Add a default class for backlash joints.

    Args:
        root: The root element of the MJCF file.
        backlash: The amount of backlash to add.
        backlash_damping: The damping value for backlash joints.
    """
    # Check if default element exists
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")

    # Check if backlash class already exists
    backlash_default = default.find(".//default[@class='backlash']")
    if backlash_default is None:
        backlash_default = ET.SubElement(default, "default", attrib={"class": "backlash"})

        # Add joint properties for backlash
        # These values are based on the reference example
        ET.SubElement(
            backlash_default,
            "joint",
            attrib={
                "damping": f"{backlash_damping:.6g}",
                "frictionloss": "0",
                "armature": "0.01",
                "limited": "true",
                "range": f"{-backlash:.6g} {backlash:.6g}",
            },
        )

        logger.info("Added backlash default class")


def find_parent_body(joint: ET.Element, root: ET.Element) -> ET.Element | None:
    """Find the parent body element of a joint.

    Args:
        joint: The joint element.
        root: The root element of the MJCF file.

    Returns:
        The parent body element, or None if not found.
    """
    # Find all body elements
    for body in root.findall(".//body"):
        # Check if the joint is a direct child of this body
        for child in body:
            if child == joint:
                return body
    return None


def add_backlash_joints(root: ET.Element) -> None:
    """Add backlash joints to all hinge joints.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all hinge joints
    hinge_joints = root.findall(".//joint[@type='hinge']")

    for joint in hinge_joints:
        # Skip if this is already a backlash joint
        if joint.get("class") == "backlash":
            continue

        # Get the joint name and parent body
        joint_name = joint.get("name")
        if joint_name is None:
            continue

        parent_body = find_parent_body(joint, root)
        if parent_body is None:
            continue

        # Create a backlash joint with the same axis
        backlash_joint = ET.Element(
            "joint",
            attrib={"name": f"{joint_name}_backlash", "class": "backlash"},
        )

        # Copy the axis attribute if it exists
        if (axis := joint.get("axis")) is not None:
            backlash_joint.set("axis", axis)

        # Copy the joint position and quaternion.
        if (pos := joint.get("pos")) is not None:
            backlash_joint.set("pos", pos)
        if (quat := joint.get("quat")) is not None:
            backlash_joint.set("quat", quat)

        # Add the backlash joint after the original joint
        # Find the index of the joint in the parent body's children
        joint_index = -1
        for i, child in enumerate(parent_body):
            if child == joint:
                joint_index = i
                break

        if joint_index >= 0:
            parent_body.insert(joint_index + 1, backlash_joint)
            logger.info("Added backlash joint for %s", joint_name)


def add_backlash(mjcf_path: str | Path, backlash: float, backlash_damping: float) -> None:
    """Add backlash joints to hinge joints in the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
        backlash: The amount of backlash to add.
        backlash_damping: The damping value for backlash joints.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    add_backlash_default(root, backlash, backlash_damping)
    add_backlash_joints(root)
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    parser.add_argument("backlash", type=float)
    parser.add_argument("--damping", type=float, default=0.01)
    args = parser.parse_args()

    add_backlash(args.mjcf_path, args.backlash, args.damping)


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.add_backlash
    main()
