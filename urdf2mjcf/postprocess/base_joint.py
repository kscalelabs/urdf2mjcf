"""Defines a post-processing function that handles base joints correctly."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_base_joint(mjcf_path: str | Path) -> None:
    """Fixes the base joint configuration.

    Args:
        mjcf_path: Path to the MJCF file
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    # Find the robot root body
    robot_body = worldbody.find("body")
    if robot_body is None:
        logger.warning("No robot body found in worldbody")
        return

    has_joint = robot_body.find("joint") is not None

    if has_joint:
        logger.warning("Robot body already has a joint; adding a new root body")

        # Create new root body with free joint
        new_root = ET.Element(
            "body",
            attrib={
                "name": "root",
                "pos": robot_body.get("pos", "0 0 0"),
                "quat": robot_body.get("quat", "1 0 0 0"),
            },
        )
        ET.SubElement(new_root, "freejoint", attrib={"name": "floating_base"})

        # Move robot body under new root
        worldbody.remove(robot_body)
        robot_body.attrib["pos"] = "0 0 0"  # Reset position relative to new root
        robot_body.attrib["quat"] = "1 0 0 0"  # Reset orientation relative to new root
        new_root.append(robot_body)
        worldbody.append(new_root)

    else:
        logger.warning("Robot body does not have a joint; adding a freejoint")
        robot_body.insert(0, ET.Element("freejoint", attrib={"name": "floating_base"}))

    # Save changes
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    fix_base_joint(args.mjcf_path)


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.base_joint
    main()
