"""Adds explicit floor contact pairs for specified links.

This script adds explicit contact pairs between specified links' collision boxes
and the floor.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def add_explicit_floor_contacts(
    mjcf_path: str | Path,
    contact_links: Sequence[str],
    class_name: str = "collision",
    floor_name: str = "floor",
) -> None:
    """Adds explicit floor contact pairs for specified links.

    For each specified link, this function finds its collision box geom and adds
    an explicit contact pair between that geom and the floor in the MJCF's contact
    section.

    Args:
        mjcf_path: Path to the MJCF file.
        contact_links: List of link (body) names to add floor contacts for.
        class_name: The class name of the collision geoms to use.
        floor_name: The name of the floor geom to contact with.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Create or get the contact section
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    contact_link_set = set(contact_links)
    geom_names = []

    # Find all the collision box geoms for the specified links
    for body_elem in root.iter("body"):
        body_name = body_elem.attrib.get("name", "")
        if body_name not in contact_link_set:
            continue
        contact_link_set.remove(body_name)

        # Find the box geom in the body with the specified class
        box_geoms = [
            geom
            for geom in body_elem.findall("geom")
            if geom.attrib.get("type", "").lower() == "box"
            and geom.attrib.get("class", "").lower() == class_name.lower()
        ]

        if len(box_geoms) == 0:
            logger.warning(f"No box geom with class {class_name} found in link {body_name}")
            continue
        if len(box_geoms) > 1:
            logger.warning(f"Multiple box geoms with class {class_name} found in link {body_name}, using first one")

        box_geom = box_geoms[0]
        geom_name = box_geom.attrib.get("name")
        if not geom_name:
            logger.warning(f"Box geom in link {body_name} has no name attribute")
            continue

        geom_names.append(geom_name)

    if contact_link_set:
        logger.warning(f"Some links were not found in the MJCF file: {contact_link_set}")

    # Add contact pairs for each found geom with the floor
    for geom_name in geom_names:
        pair = ET.SubElement(contact, "pair")
        pair.attrib["geom1"] = geom_name
        pair.attrib["geom2"] = floor_name

    # Save the modified MJCF file
    save_xml(mjcf_path, tree)
    logger.info("Added explicit floor contacts to MJCF file at %s", mjcf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adds explicit floor contact pairs for specified links.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--links", nargs="+", required=True, help="List of link names to add floor contacts for.")
    parser.add_argument(
        "--class-name",
        type=str,
        default="collision",
        help="Class name of the collision geoms to use.",
    )
    parser.add_argument(
        "--floor-name",
        type=str,
        default="floor",
        help="Name of the floor geom to contact with.",
    )
    args = parser.parse_args()

    add_explicit_floor_contacts(args.mjcf_path, args.links, args.class_name, args.floor_name)


if __name__ == "__main__":
    main()
