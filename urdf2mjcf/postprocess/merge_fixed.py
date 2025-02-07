"""Defines a post-processing function that merges MJCF fixed joints into their parent body."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def remove_fixed_joints(mjcf_path: str | Path) -> None:
    """Merges fixed joints into their parent body.

    This function works by finding all of the body links which do not have a
    joint element, and converting them from a body element to a site element.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Find all body elements
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        return

    bodies_to_merge = []

    # Find all bodies that don't have joints
    for parent_body in worldbody.findall(".//body"):
        if parent_body.find("freejoint") is not None:
            continue

        for child_body in parent_body.findall("body"):
            if child_body.find("joint") is not None:
                continue

            bodies_to_merge.append((parent_body, child_body))

    for parent_body, child_body in bodies_to_merge:
        parent_name = parent_body.attrib["name"]
        child_name = child_body.attrib["name"]
        logger.info("Merging body %s into %s", child_name, parent_name)

        # Create a site element at the position of the merged body
        site = ET.SubElement(parent_body, "site")
        site.set("name", child_name)
        for attr in ["pos", "quat", "euler"]:
            if (val := child_body.get(attr)) is not None:
                site.set(attr, val)

        # Transfer all child elements except inertial to the parent
        for grandchild in child_body:
            if grandchild.tag != "inertial":
                child_body.remove(grandchild)
                parent_body.append(grandchild)

        # Remove the merged body
        parent_body.remove(child_body)

    # Save the modified XML
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    remove_fixed_joints(args.mjcf_path)


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.merge_fixed
    main()
