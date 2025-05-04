"""Converts some mesh geometries into easier-to-collide shapes.

For frameworks like MJX, it is usually a good idea to model some of the links
as simpler shapes, since it is more robust to collide them with other links in
the scene.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import mujoco
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

    # Load the MJCF model with Mujoco to get the proper transformations.
    # (This will account for any joint or body-level rotations.)
    try:
        model_mujoco = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model_mujoco)
    except Exception as e:
        logger.error("Failed to load MJCF in Mujoco: %s", e)
        raise

    # Run one step.
    mujoco.mj_step(model_mujoco, data)

    name_to_geom = {g.name: g for g in collision_geometries}
    link_set = set(name_to_geom.keys())

    # Iterate over all <body> elements and process those in foot_links.
    for body_elem in root.iter("body"):
        body_name = body_elem.attrib.get("name", "")
        if body_name not in link_set:
            continue
        link_set.remove(body_name)
        collision_geom = name_to_geom[body_name]

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
        found_visual_mesh = len(visual_meshes) == 1
        if not found_visual_mesh:
            logger.warning(
                "No visual mesh found for %s in body %s."
                "Box collision will be added, but corresponding visual will not be updated.",
                visual_mesh_name,
                body_name,
            )
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

        # find geom by name in the XML and use its attributes
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
        geom_r = R.from_quat(geom_quat).as_matrix()

        # Transform vertices to mesh-local coordinates
        local_vertices = vertices.copy()

        # Apply any local transform from the mesh geom
        if np.any(geom_pos != 0) or not np.allclose(geom_quat, [1, 0, 0, 0]):
            # Transform vertices to account for geom's local position and orientation
            local_vertices = (geom_r @ vertices.T).T + geom_pos

        match collision_geom.collision_type:
            case CollisionType.BOX:

                # Compute bounding box in local coordinates
                min_x, min_y, min_z = local_vertices.min(axis=0)
                max_x, max_y, max_z = local_vertices.max(axis=0)

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

                # Copies over any other attributes from the original mesh geom.
                for key in ("material", "class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
                    if key in mesh_geom.attrib:
                        box_geom.attrib[key] = mesh_geom.attrib[key]

                body_elem.append(box_geom)

                # Update the visual mesh to be a box instead of creating a new one
                # Replace the mesh with a box
                if found_visual_mesh:
                    visual_mesh.attrib["type"] = "box"
                    visual_mesh.attrib["pos"] = " ".join(f"{v:.6f}" for v in box_pos)
                    visual_mesh.attrib["quat"] = " ".join(f"{v:.6f}" for v in box_quat)
                    visual_mesh.attrib["size"] = " ".join(f"{v:.6f}" for v in box_size)

                    # Remove mesh attribute as it's now a box
                    if "mesh" in visual_mesh.attrib:
                        del visual_mesh.attrib["mesh"]

                    logger.info("Updated visual mesh %s to be a box", visual_mesh_name)

            case CollisionType.PARALLEL_LONG_CAPSULES:
                raise NotImplementedError("Parallel long capsules not implemented.")

            case _:
                raise NotImplementedError(f"Collision type {collision_geom.collision_type} not implemented.")

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
