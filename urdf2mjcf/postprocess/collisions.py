"""Converts some mesh geometries into easier-to-collide shapes.

For frameworks like MJX, it is usually a good idea to model some of the links
as simpler shapes, since it is more robust to collide them with other links in
the scene.
"""

import argparse
import itertools
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from urdf2mjcf.model import CollisionGeometry, CollisionType
from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def update_collisions(
    mjcf_path: str | Path,
    collision_geometries: Sequence[CollisionGeometry],
    class_name: str = "collision",
) -> None:
    """Converts some mesh geometries into easier-to-collide shapes.

    Args:
        mjcf_path: Path to the MJCF file.
        collision_geometries: List of collision geometries to process.
        class_name: The class name to use for the sphere geoms.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Get all the meshes from the <asset> element.
    asset = root.find("asset")
    if asset is None:
        raise ValueError("No <asset> element found in the MJCF file.")
    meshes = asset.findall("mesh")
    mesh_name_to_path = {
        mesh.attrib.get("name", mesh.attrib.get("file", "MISSING")): mesh.attrib["file"] for mesh in meshes
    }

    name_to_geom = {g.name: g for g in collision_geometries}
    link_set = set(name_to_geom.keys())

    # Iterate over all <body> elements and process those in foot_links.
    for body_elem in root.iter("body"):
        body_name = body_elem.attrib.get("name", "")
        if body_name not in link_set:
            continue
        link_set.remove(body_name)
        collision_geom_info = name_to_geom[body_name]

        # Find the mesh geom in the body, disambiguating by class if necessary.
        mesh_geoms = [geom for geom in body_elem.findall("geom") if geom.attrib.get("type", "").lower() == "mesh"]
        if len(mesh_geoms) == 0:
            raise ValueError(f"No mesh geom found in link {body_name}")
        if len(mesh_geoms) > 1:
            logger.warning("Got multiple mesh geoms in link %s; attempting to use class %s", body_name, class_name)
            mesh_geoms = [geom for geom in mesh_geoms if geom.attrib.get("class", "").lower() == class_name]

            if len(mesh_geoms) == 0:
                raise ValueError(f"No mesh geom with class {class_name} found in link {body_name}")
            if len(mesh_geoms) > 1:
                raise ValueError(f"Got multiple mesh geoms with class {class_name} in link {body_name}")

        mesh_geom = mesh_geoms[0]
        mesh_geom_name = mesh_geom.attrib.get("name")

        # Find any visual meshes in this body to get material from - using naming convention
        visual_mesh_name = f"{body_name}_visual"
        visual_meshes = [geom for geom in body_elem.findall("geom") if geom.attrib.get("name") == visual_mesh_name]
        if len(visual_meshes) == 0:
            logger.warning(
                "No visual mesh found for %s in body %s."
                "Box collision will be added, but corresponding visual will not be updated.",
                visual_mesh_name,
                body_name,
            )
            visual_mesh = None
        else:
            visual_mesh = visual_meshes[0]

        mesh_name = mesh_geom.attrib.get("mesh")
        if not mesh_name:
            logger.warning("Mesh geom in link %s does not specify a mesh file; skipping.", body_name)
            continue

        if mesh_name not in mesh_name_to_path:
            logger.warning("Mesh name %s not found in <asset> element; skipping.", mesh_name)
            continue
        mesh_file = mesh_name_to_path[mesh_name]

        # Load the mesh using trimesh.
        mesh_path = (mjcf_path.parent / mesh_file).resolve()
        try:
            mesh = trimesh.load(mesh_path)
        except Exception as e:
            logger.error("Failed to load mesh from %s for link %s: %s", mesh_path, body_name, e)
            continue

        if not isinstance(mesh, trimesh.Trimesh):
            logger.warning("Loaded mesh from %s is not a Trimesh for link %s; skipping.", mesh_path, body_name)
            continue

        # Transform the mesh vertices to world coordinates.
        vertices = mesh.vertices  # shape (n,3)

        # Find geom by name in the XML and use its attributes
        geom_pos = np.zeros(3, dtype=np.float64)
        geom_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Default identity quaternion

        # Get position and orientation from the mesh geom XML
        if "pos" in mesh_geom.attrib:
            pos_values = [float(v) for v in mesh_geom.attrib["pos"].split()]
            geom_pos[:] = pos_values  # Update values in-place

        if "quat" in mesh_geom.attrib:
            quat_values = [float(v) for v in mesh_geom.attrib["quat"].split()]
            geom_quat[:] = quat_values  # Update values in-place

        # Get rotation matrix from quaternion
        geom_r = R.from_quat(geom_quat, scalar_first=True)

        # Transform vertices to mesh-local coordinates
        local_vertices = vertices.copy()

        # Transform vertices to account for geom's local position and orientation
        local_vertices = geom_r.apply(vertices) + geom_pos

        # Compute bounding box in local coordinates
        min_x, min_y, min_z = local_vertices.min(axis=0)
        max_x, max_y, max_z = local_vertices.max(axis=0)

        match collision_geom_info.collision_type:
            case CollisionType.BOX:
                # Create box with same dimensions as original mesh bounding box
                box_size = np.array(
                    [
                        (max_x - min_x) / 2,
                        (max_y - min_y) / 2,
                        (max_z - min_z) / 2,
                    ]
                )

                # Position at center of bounding box
                box_pos = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2])

                # Use the original geom's orientation
                box_quat = geom_quat

                # Add a bounding box geom.
                box_geom = ET.Element("geom")
                box_geom.attrib["name"] = f"{mesh_geom_name}_box"
                box_geom.attrib["type"] = "box"
                box_geom.attrib["pos"] = " ".join(f"{v:.6f}" for v in box_pos)
                box_geom.attrib["quat"] = " ".join(f"{v:.6f}" for v in box_quat)
                box_geom.attrib["size"] = " ".join(f"{v:.6f}" for v in box_size)
                box_geom.attrib["material"] = "collision_material"

                # Copies over any other attributes from the original mesh geom.
                for key in ("class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
                    if key in mesh_geom.attrib:
                        box_geom.attrib[key] = mesh_geom.attrib[key]

                body_elem.append(box_geom)

                # Update the visual mesh to be a box instead of creating a new one
                # Replace the mesh with a box
                if visual_mesh is not None:
                    visual_mesh.attrib["type"] = "box"
                    visual_mesh.attrib["pos"] = " ".join(f"{v:.6f}" for v in box_pos)
                    visual_mesh.attrib["quat"] = " ".join(f"{v:.6f}" for v in box_quat)
                    visual_mesh.attrib["size"] = " ".join(f"{v:.6f}" for v in box_size)

                    # Remove mesh attribute as it's now a box
                    if "mesh" in visual_mesh.attrib:
                        del visual_mesh.attrib["mesh"]

                    logger.info("Updated visual mesh %s to be a box", visual_mesh_name)

            case CollisionType.PARALLEL_CAPSULES:
                rad = collision_geom_info.sphere_radius
                ax = collision_geom_info.axis_order
                flip_axis = collision_geom_info.flip_axis

                pairs = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

                # Determine which direction is the longest
                length = [max_x - min_x, max_y - min_y, max_z - min_z][ax[0]]

                # Use the original geom's orientation
                sphere_quat = geom_quat

                for i, min_side in enumerate((True, False)):
                    min_s, max_s = pairs[ax[1]]
                    range_s = (min_s, min_s + rad * 2) if min_side else (max_s - rad * 2, max_s)

                    mask = (local_vertices[:, ax[1]] >= range_s[0]) & (local_vertices[:, ax[1]] <= range_s[1])
                    min_v = np.min(local_vertices[mask, ax[0]])
                    max_v = np.max(local_vertices[mask, ax[0]])

                    capsule_fromto = np.zeros(6, dtype=np.float64)
                    capsule_fromto[ax[0]] = min_v + rad
                    capsule_fromto[ax[0] + 3] = max_v - rad

                    val_1 = pairs[ax[1]][0] + rad if min_side else pairs[ax[1]][1] - rad
                    capsule_fromto[ax[1]] = val_1
                    capsule_fromto[ax[1] + 3] = val_1

                    val_2 = pairs[ax[2]][0] + rad if flip_axis else pairs[ax[2]][1] - rad
                    capsule_fromto[ax[2]] = val_2
                    capsule_fromto[ax[2] + 3] = val_2

                    # Create the capsule geom
                    capsule_geom = ET.Element("geom")
                    capsule_geom.attrib["name"] = f"{mesh_geom_name}_capsule_{i}"
                    capsule_geom.attrib["type"] = "capsule"
                    capsule_geom.attrib["quat"] = " ".join(f"{v:.6f}" for v in sphere_quat)
                    capsule_geom.attrib["fromto"] = " ".join(f"{v:.6f}" for v in capsule_fromto)
                    capsule_geom.attrib["size"] = f"{rad:.6f} {length / 2:.6f}"
                    capsule_geom.attrib["material"] = "collision_material"

                    # Copy over any other attributes from the original mesh geom
                    for key in ("class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
                        if key in mesh_geom.attrib:
                            capsule_geom.attrib[key] = mesh_geom.attrib[key]

                    body_elem.append(capsule_geom)

                    # Update the visual mesh to be capsules instead of creating new ones
                    if visual_mesh is not None:
                        visual_capsule = ET.Element("geom")
                        visual_capsule.attrib["name"] = f"{visual_mesh_name}_capsule_{i}"
                        visual_capsule.attrib["type"] = "capsule"
                        visual_capsule.attrib["quat"] = " ".join(f"{v:.6f}" for v in sphere_quat)
                        visual_capsule.attrib["fromto"] = " ".join(f"{v:.6f}" for v in capsule_fromto)
                        visual_capsule.attrib["size"] = f"{rad:.6f} {length / 2:.6f}"

                        # Copy over material and class attributes
                        for key in ("material", "class"):
                            if key in visual_mesh.attrib:
                                visual_capsule.attrib[key] = visual_mesh.attrib[key]

                        body_elem.append(visual_capsule)

                if visual_mesh is not None:
                    body_elem.remove(visual_mesh)
                    logger.info("Updated visual mesh %s to be capsules", visual_mesh_name)

            case CollisionType.CORNER_SPHERES:
                rad = collision_geom_info.sphere_radius
                ax = collision_geom_info.axis_order
                flip_axis = collision_geom_info.flip_axis

                pairs = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

                # Use the original geom's orientation
                sphere_quat = geom_quat

                for i, (min_x, min_y) in enumerate(itertools.product((True, False), (True, False))):
                    x = pairs[ax[0]][0] + rad if min_x else pairs[ax[0]][1] - rad
                    y = pairs[ax[1]][0] + rad if min_y else pairs[ax[1]][1] - rad
                    z = pairs[ax[2]][0] + rad if flip_axis else pairs[ax[2]][1] - rad

                    xyz = [0, 0, 0]
                    xyz[ax[0]] = x
                    xyz[ax[1]] = y
                    xyz[ax[2]] = z
                    x, y, z = xyz

                    # Create the capsule geom
                    sphere_geom = ET.Element("geom")
                    sphere_geom.attrib["name"] = f"{mesh_geom_name}_sphere_{i}"
                    sphere_geom.attrib["type"] = "sphere"
                    sphere_geom.attrib["quat"] = " ".join(f"{v:.6f}" for v in sphere_quat)
                    sphere_geom.attrib["pos"] = " ".join(f"{v:.6f}" for v in (x, y, z))
                    sphere_geom.attrib["size"] = f"{rad:.6f}"
                    sphere_geom.attrib["material"] = "collision_material"

                    # Copy over any other attributes from the original mesh geom
                    for key in ("class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
                        if key in mesh_geom.attrib:
                            sphere_geom.attrib[key] = mesh_geom.attrib[key]

                    body_elem.append(sphere_geom)

                    # Update the visual mesh to be capsules instead of creating new ones
                    if visual_mesh is not None:
                        visual_sphere = ET.Element("geom")
                        visual_sphere.attrib["name"] = f"{visual_mesh_name}_sphere_{i}"
                        visual_sphere.attrib["type"] = "sphere"
                        visual_sphere.attrib["quat"] = " ".join(f"{v:.6f}" for v in sphere_quat)
                        visual_sphere.attrib["pos"] = " ".join(f"{v:.6f}" for v in (x, y, z))
                        visual_sphere.attrib["size"] = f"{rad:.6f}"

                        # Copy over material and class attributes
                        for key in ("material", "class"):
                            if key in visual_mesh.attrib:
                                visual_sphere.attrib[key] = visual_mesh.attrib[key]

                        body_elem.append(visual_sphere)

                if visual_mesh is not None:
                    body_elem.remove(visual_mesh)
                    logger.info("Updated visual mesh %s to be corner spheres", visual_mesh_name)

            case CollisionType.SINGLE_SPHERE:
                rad = collision_geom_info.sphere_radius
                ax = collision_geom_info.axis_order
                flip_axis = collision_geom_info.flip_axis
                pairs = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

                min_v, max_v = pairs[ax[2]]
                pairs[ax[2]] = (min_v + rad, min_v + rad) if flip_axis else (max_v - rad, max_v - rad)

                sphere_pos = np.array([sum(pairs[0]) / 2, sum(pairs[1]) / 2, sum(pairs[2]) / 2])
                sphere_quat = geom_quat

                sphere_geom = ET.Element("geom")
                sphere_geom.attrib["name"] = f"{mesh_geom_name}_sphere"
                sphere_geom.attrib["type"] = "sphere"
                sphere_geom.attrib["quat"] = " ".join(f"{v:.6f}" for v in sphere_quat)
                sphere_geom.attrib["pos"] = " ".join(f"{v:.6f}" for v in sphere_pos)
                sphere_geom.attrib["size"] = f"{rad:.6f}"
                sphere_geom.attrib["material"] = "collision_material"

                body_elem.append(sphere_geom)

                if visual_mesh is not None:
                    visual_mesh.attrib["type"] = "sphere"
                    visual_mesh.attrib["pos"] = " ".join(f"{v:.6f}" for v in sphere_pos)
                    visual_mesh.attrib["quat"] = " ".join(f"{v:.6f}" for v in sphere_quat)
                    visual_mesh.attrib["size"] = f"{rad:.6f}"

                    # Remove mesh attribute as it's now a sphere
                    if "mesh" in visual_mesh.attrib:
                        del visual_mesh.attrib["mesh"]

                    logger.info("Updated visual mesh %s to be a single sphere", visual_mesh_name)

            case _:
                raise NotImplementedError(f"Collision type {collision_geom_info.collision_type} not implemented.")

        # Remove the original mesh geom from the body.
        body_elem.remove(mesh_geom)

    if link_set:
        raise ValueError(f"Found {len(link_set)} collision geometries that were not found in the MJCF file: {link_set}")

    # Save the modified MJCF file.
    save_xml(mjcf_path, tree)
    logger.info("Saved modified MJCF file with feet converted to boxes at %s", mjcf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts MJCF collision geometries from meshes to boxes.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--links", nargs="+", required=True, help="List of link names to convert.")
    args = parser.parse_args()

    collision_geometries = [CollisionGeometry(name=name, collision_type=CollisionType.BOX) for name in args.links]
    update_collisions(args.mjcf_path, collision_geometries)


if __name__ == "__main__":
    main()
