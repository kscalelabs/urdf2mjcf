"""Defines a post-processing function that combines duplicate materials."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def remove_redundant_materials(root: ET.Element) -> None:
    """Removes redundant materials from the MJCF file.

    Args:
        root: The root element of the MJCF file.
    """
    # Find all materials in the asset section
    asset = root.find("asset")
    if asset is None:
        return

    materials = asset.findall("material")
    if not materials:
        return

    # Group materials by their RGBA values
    rgba_to_materials: dict[str, list[ET.Element]] = {}
    for material in materials:
        rgba = material.get("rgba")
        if rgba:
            rgba_to_materials.setdefault(rgba, []).append(material)

    # For each group of duplicate materials, keep the first one and update references
    for rgba, duplicate_materials in rgba_to_materials.items():
        if len(duplicate_materials) <= 1:
            continue

        # Keep the first material and remove others
        kept_material = duplicate_materials[0]
        if (kept_name := kept_material.get("name")) is None:
            continue

        for material in duplicate_materials[1:]:
            duplicate_name = material.get("name")
            logger.info("Replacing material %s with %s", duplicate_name, kept_name)

            # Update all geom references to this material
            for geom in root.findall(".//geom[@material='{}']".format(duplicate_name)):
                geom.set("material", kept_name)

            # Remove the duplicate material
            asset.remove(material)


def is_close_to_identity(value: str, tolerance: float = 1e-6) -> bool:
    """Checks if a space-separated string of numbers is close to identity.

    Args:
        value: Space-separated string of numbers
        tolerance: Maximum allowed deviation from identity

    Returns:
        True if values are close to identity values
    """
    try:
        nums = [float(x) for x in value.split()]
        if len(nums) == 3:  # pos
            return all(abs(x) <= tolerance for x in nums)
        elif len(nums) == 4:  # quat
            return abs(nums[0] - 1) <= tolerance and all(abs(x) <= tolerance for x in nums[1:])
        return False
    except (ValueError, TypeError):
        return False


def remove_redundant_geoms(root: ET.Element) -> None:
    """Removes redundant geoms from the MJCF file.

    We can remove redundant `pos` tags which are close to [0, 0, 0] and
    redundant `quat` tags which are close to [1, 0, 0, 0].

    Args:
        root: The root element of the MJCF file.
    """
    # Find all elements with pos or quat attributes
    for elem in root.findall(".//*[@pos]"):
        if is_close_to_identity(elem.attrib["pos"]):
            del elem.attrib["pos"]

    for elem in root.findall(".//*[@quat]"):
        if is_close_to_identity(elem.attrib["quat"]):
            del elem.attrib["quat"]

    # Count removed attributes for logging
    removed_pos = len(root.findall(".//*[@pos]"))
    removed_quat = len(root.findall(".//*[@quat]"))

    if removed_pos > 0 or removed_quat > 0:
        logger.info(
            "Removed %d redundant pos and %d redundant quat attributes",
            removed_pos,
            removed_quat,
        )


def remove_redundancies(mjcf_path: str | Path) -> None:
    """Remove redundancies from the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    remove_redundant_materials(root)
    remove_redundant_geoms(root)

    # Save the modified MJCF file
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    remove_redundancies(args.mjcf_path)


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.remove_redundancies
    main()
