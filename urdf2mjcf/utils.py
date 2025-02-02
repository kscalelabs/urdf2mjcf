"""Defines utility functions."""

import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator
from xml.dom import minidom


def iter_meshes(
    urdf_path: Path,
) -> Iterator[
    tuple[
        tuple[ET.Element, Path] | tuple[None, None],
        tuple[ET.Element, Path] | tuple[None, None],
    ],
]:
    """Iterate over mesh elements in a URDF file.

    Args:
        urdf_path: Path to the URDF file.

    Yields:
        A tuple of ((visual_element, mesh_path), (collision_element, mesh_path))
        where either tuple may be (None, None) if no mesh is present.
    """
    tree = ET.parse(urdf_path)
    urdf_dir = urdf_path.parent

    for link in tree.findall("link"):
        visual_mesh: tuple[ET.Element, Path] | tuple[None, None] = (None, None)
        collision_mesh: tuple[ET.Element, Path] | tuple[None, None] = (None, None)

        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None and "filename" in mesh.attrib:
                    mesh_path = urdf_dir / mesh.attrib["filename"]
                    visual_mesh = (mesh, mesh_path)

        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None and "filename" in mesh.attrib:
                    mesh_path = urdf_dir / mesh.attrib["filename"]
                    collision_mesh = (mesh, mesh_path)

        if visual_mesh != (None, None) or collision_mesh != (None, None):
            yield visual_mesh, collision_mesh


def save_xml(path: str | Path | io.StringIO, tree: ET.ElementTree | ET.Element) -> None:
    if isinstance(tree, ET.ElementTree):
        tree = tree.getroot()
    xmlstr = minidom.parseString(ET.tostring(tree)).toprettyxml(indent="  ")
    xmlstr = re.sub(r"\n\s*\n", "\n", xmlstr)

    # Add newlines between second-level nodes
    root = ET.fromstring(xmlstr)
    for child in root[:-1]:
        child.tail = "\n\n  "
    xmlstr = ET.tostring(root, encoding="unicode")

    if isinstance(path, io.StringIO):
        path.write(xmlstr)
    else:
        with open(path, "w") as f:
            f.write(xmlstr)
