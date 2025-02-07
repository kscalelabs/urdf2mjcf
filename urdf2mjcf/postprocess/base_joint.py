"""Defines a post-processing function that handles base joints correctly."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_base_joint(mjcf_path: str | Path, floating_base: bool = False) -> None:
    """Fixes the base joint configuration.

    If floating_base is True, creates a new root body with a free joint.
    Otherwise, keeps the original root body configuration.

    Args:
        mjcf_path: Path to the MJCF file
        floating_base: Whether to make the base floating
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    # Find the robot root body
    robot_body = None
    for body in worldbody.findall("body"):
        if "floating_base" in [j.get("name", "") for j in body.findall("joint")]:
            robot_body = body
            break

    if robot_body is None:
        return

    # Remove existing floating base joint if any
    for joint in robot_body.findall("joint"):
        if joint.get("name") == "floating_base":
            robot_body.remove(joint)

    if floating_base:
        # Create new root body with free joint
        new_root = ET.Element(
            "body",
            attrib={"name": "root", "pos": robot_body.get("pos", "0 0 0"), "quat": robot_body.get("quat", "1 0 0 0")},
        )
        ET.SubElement(new_root, "freejoint", attrib={"name": "floating_base"})

        # Move robot body under new root
        worldbody.remove(robot_body)
        robot_body.attrib["pos"] = "0 0 0"  # Reset position relative to new root
        robot_body.attrib["quat"] = "1 0 0 0"  # Reset orientation relative to new root
        new_root.append(robot_body)
        worldbody.append(new_root)

    # Save changes
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
