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
            "group": "0",  # Default value for group
            "type": "plane",
            "size": "0 0 0.05",
            "material": "groundplane",
            "rgba": "1 1 1 0.3",  # Make transparent
        }

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
    floor_geom.attrib["size"] = "0 0 0.05"
    worldbody.append(floor_geom)


def add_floor_assets(root: ET.Element) -> None:
    """Add the needed assets for the floor.

    Args:
        root: The root element of the MJCF file.
    """
    # Find or create asset element
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add texture for groundplane
    _ = ET.SubElement(
        asset,
        "texture",
        attrib={
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )

    # Add material for groundplane
    _ = ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )


def add_floor(mjcf_path: str | Path, floor_name: str = "floor") -> None:
    """Add a floor to the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
        floor_name: The name of the floor.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    add_floor_assets(root)
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
