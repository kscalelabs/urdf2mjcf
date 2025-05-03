"""Replaces foot mesh geoms with two parallel capsules along the local x-axis.

For each specified foot link (body), this script finds a mesh geom
(optionally matching `class_name`). It calculates the mesh's bounding box
in its local frame (relative to the body, considering the geom's transform).
Based on this bounding box, it creates two capsule geoms oriented along the
local X-axis, positioned symmetrically along the local Y-axis. The original
mesh geom is removed.
"""

import argparse
import copy
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from urdf2mjcf.utils import save_xml  # Assuming this utility exists

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Add basic logging configuration


def add_new_feet_collision(
    mjcf_path: str | Path,
    foot_links: Sequence[str],
    class_name: str = "collision",
    min_radius: float = 0.005,  # Add minimum radius to avoid zero-size capsules
) -> None:
    """Replaces foot mesh geoms with two parallel capsules along the local x-axis.

    Args:
        mjcf_path: Path to the MJCF file.
        foot_links: List of link (body) names containing foot meshes to process.
        class_name: The class name used to identify the collision mesh geom
            if multiple mesh geoms exist in a body.
        min_radius: Minimum radius for the capsules to prevent zero-size geoms.
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

    foot_link_set = set(foot_links)

    # Iterate over all <body> elements and process those in foot_links.
    for body_elem in root.iter("body"):
        body_name = body_elem.attrib.get("name", "")
        if body_name not in foot_link_set:
            continue
        foot_link_set.remove(body_name)

        mesh_geoms = [geom for geom in body_elem.findall("geom") if geom.attrib.get("type", "").lower() == "mesh"]

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
        vertices = mesh.vertices  # shape (n, 3)

        # Get local transform attributes from the mesh geom XML
        geom_pos = np.zeros(3, dtype=np.float64)
        geom_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

        # Get position and orientation from the mesh geom XML
        if "pos" in mesh_geom.attrib:
            pos_values = [float(v) for v in mesh_geom.attrib["pos"].split()]
            geom_pos[:] = pos_values  # Update values in-place

        if "quat" in mesh_geom.attrib:
            quat_values = [float(v) for v in mesh_geom.attrib["quat"].split()]
            geom_quat[:] = quat_values  # Update values in-place

        # Transform vertices into the geom's local frame (relative to the body)
        # This frame accounts for the geom's pos and quat offset from the body origin
        geom_rot_mat = R.from_quat(geom_quat[[1, 2, 3, 0]]).as_matrix()  # Scipy uses x, y, z, w
        local_vertices = (geom_rot_mat @ vertices.T).T + geom_pos

        # Compute axis-aligned bounding box (AABB) in this local frame
        min_coords = local_vertices.min(axis=0)
        max_coords = local_vertices.max(axis=0)
        min_x, min_y, min_z = min_coords
        max_x, max_y, max_z = max_coords

        # Define Capsule Parameters based on AABB dimensions
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0

        # Dimensions of the AABB
        dim_z = max_z - min_z  # assumed thickness

        # Capsule radius based on thickness (Z-dimension of AABB)
        capsule_radius = max(dim_z / 2.0, min_radius)

        # Capsule axis runs along length (X-dimension of AABB)
        axis_start_x = min_x
        axis_end_x = max_x

        y_coord = center_y

        # Assume half the thickness for the capsule
        z_offset = dim_z / 4.0
        z_coord_lower = center_z - z_offset
        z_coord_upper = center_z + z_offset

        fromto_left = (
            f"{axis_start_x:.6f} {y_coord:.6f} {z_coord_lower:.6f} {axis_end_x:.6f} {y_coord:.6f} {z_coord_lower:.6f}"
        )
        fromto_right = (
            f"{axis_start_x:.6f} {y_coord:.6f} {z_coord_upper:.6f} {axis_end_x:.6f} {y_coord:.6f} {z_coord_upper:.6f}"
        )

        # Create left capsule
        capsule_left = ET.Element("geom")
        capsule_left.attrib["name"] = f"{mesh_geom_name}_capsule_left"
        capsule_left.attrib["fromto"] = fromto_left

        # Copies over any other attributes from the original mesh geom.
        for key in ("material", "class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
            if key in mesh_geom.attrib:
                capsule_left.attrib[key] = mesh_geom.attrib[key]
        body_elem.append(capsule_left)

        # Create right capsule
        capsule_right = ET.Element("geom")
        capsule_right.attrib["name"] = f"{mesh_geom_name}_capsule_right"
        capsule_right.attrib["fromto"] = fromto_right

        # Copies over any other attributes from the original mesh geom.
        for key in ("material", "class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
            if key in mesh_geom.attrib:
                capsule_left.attrib[key] = mesh_geom.attrib[key]
        body_elem.append(capsule_right)

        if found_visual_mesh:
            visual_mesh.attrib["type"] = "capsule"
            visual_mesh.attrib["size"] = f"{capsule_radius:.6f}"
            visual_mesh.attrib["fromto"] = fromto_left
            visual_mesh.attrib["name"] = f"{mesh_geom_name}_capsule_left_visual"

            visual_mesh_right = copy.deepcopy(visual_mesh)
            visual_mesh_right.attrib["fromto"] = fromto_right
            visual_mesh_right.attrib["name"] = f"{mesh_geom_name}_capsule_right_visual"
            body_elem.append(visual_mesh_right)
            logger.info("Updated visual mesh %s to be two capsules", visual_mesh_name)

            if "mesh" in visual_mesh.attrib:
                del visual_mesh.attrib["mesh"]

        body_elem.remove(mesh_geom)

    save_xml(mjcf_path, tree)
    logger.info(f"Saved modified MJCF file with feet converted to capsules at {mjcf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts MJCF feet from meshes to boxes.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--links", nargs="+", required=True, help="List of link names to convert into foot boxes.")
    parser.add_argument(
        "--class_name",
        type=str,
        default="collision",
        help="Class name used to identify the correct mesh geom if multiple meshes exist in a link.",
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=0.005,
        help="Minimum radius for the generated capsules to avoid zero-size geoms.",
    )
    args = parser.parse_args()

    add_new_feet_collision(args.mjcf_path, args.links, args.class_name, args.min_radius)


if __name__ == "__main__":
    main()
