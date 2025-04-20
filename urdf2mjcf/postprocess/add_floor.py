"""Defines a post-processing function that adds a floor to the Mujoco model.

This script adds a floor to the MJCF file as either a plane or a height field (hfield).
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def add_floor_default(root: ET.Element, floor_name: str = "floor") -> None:
    """Add a default class for the floor.

    Args:
        root: The root element of the MJCF file.
        floor_name: The name of the floor class.
    """
    # Check if default element exists
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")

    # Check if floor class already exists
    floor_default = default.find(f".//default[@class='{floor_name}']")
    if floor_default is None:
        floor_default = ET.SubElement(default, "default", attrib={"class": floor_name})

        # Add geom properties for floor
        geom_attrib = {
            "contype": "1",  # Enable collision
            "conaffinity": "1",  # Enable collision with all objects
            "group": "1",  # Group 1 for floor
            "friction": "0.8 0.02 0.01",  # Default friction values
            "condim": "6",  # Default contact dimension
        }

        geom_attrib["type"] = "plane"
        geom_attrib["size"] = "100 100 0.1"  # Large plane with small thickness

        ET.SubElement(floor_default, "geom", attrib=geom_attrib)


def add_floor_geom(root: ET.Element, floor_name: str = "floor") -> None:
    """Add a floor geom to the worldbody.

    Args:
        root: The root element of the MJCF file.
        floor_name: The name of the floor geom.
    """
    # Find the worldbody element
    worldbody = root.find("worldbody")
    if worldbody is None:
        logger.warning("No worldbody element found in the MJCF file.")
        return

    # Check if floor already exists
    existing_floor = worldbody.find(f".//geom[@name='{floor_name}']")
    if existing_floor is not None:
        logger.info(f"Floor '{floor_name}' already exists in the MJCF file.")
        return

    # Create the floor geom
    floor_geom = ET.Element("geom")
    floor_geom.attrib["name"] = floor_name
    floor_geom.attrib["class"] = floor_name
    floor_geom.attrib["pos"] = "0 0 0"  # Position at origin
    floor_geom.attrib["quat"] = "1 0 0 0"  # Identity quaternion (no rotation)

    # Add the floor geom to the worldbody
    worldbody.append(floor_geom)


def add_floor(mjcf_path: str | Path, floor_name: str = "floor") -> None:
    """Add a floor to the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
        floor_name: The name of the floor.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    add_floor_default(root, floor_name)
    add_floor_geom(root, floor_name)
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adds a floor to the MJCF model.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument(
        "--name",
        type=str,
        default="floor",
        help="Name of the floor.",
    )
    args = parser.parse_args()

    add_floor(args.mjcf_path, args.name)


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.add_floor
    main()
