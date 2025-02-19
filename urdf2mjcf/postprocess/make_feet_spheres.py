"""Converts the feet of an MJCF model into spheres.

For each specified foot link (which is assumed to contain a mesh geom),
this script loads the MJCF both with XML (for modification) and with Mujoco
(for computing the correct transformation of the mesh geometry). For each
foot link, it loads the mesh file, applies the transform computed by Mujoco
(i.e. the combined effect of any joint, body, and geom transformations),
computes the axis-aligned bounding box in world coordinates, finds the bottom
four corners of that bounding box (with the provided sphere radius), converts
these points into the body-local coordinates, creates sphere geoms at each
location, and finally removes the original mesh geom.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def convert_feet_to_spheres(
    mjcf_path: str | Path,
    foot_links: list[str],
    sphere_radius: float,
    class_name: str = "collision",
) -> None:
    """Converts the feet of an MJCF model into spheres using Mujoco.

    For each specified foot link, this function loads the MJCF file both as an
    XML tree (for later writing) and as a Mujoco model to obtain the correct
    (world) transformation for the mesh geom. It then loads the mesh file,
    transforms its vertices using Mujoco's computed geom transform, computes
    its axis-aligned bounding box in world coordinates, extracts the bottom
    four corners (with z-coordinate at the minimum), converts these positions
    into the body-local frame, creates sphere geoms at those locations (with
    the provided sphere radius), and finally removes the original mesh geom.

    Args:
        mjcf_path: Path to the MJCF file.
        foot_links: List of link (body) names to process.
        sphere_radius: The sphere radius (in meters) to use.
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

    # Iterate over all <body> elements and process those in foot_links.
    for body_elem in root.iter("body"):
        body_name = body_elem.attrib.get("name", "")
        if body_name not in foot_links:
            continue

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

        # Convert the world vertices to the body-local coordinate system.
        # This ensures that the "bottom" of the mesh corresponds to the minimal z-value in body coordinates.
        body = data.body(body_name)

        # Since the body rotation matrix is orthogonal, its inverse is its transpose.
        body_r = body.xmat.reshape(3, 3)
        body_r_inv = body_r.T
        local_vertices = (body_r_inv @ vertices.T).T

        # Gets the bounding box of the hull.
        min_x, min_y, min_z = local_vertices.min(axis=0)
        max_x, max_y, max_z = local_vertices.max(axis=0)

        # The bottom face in body coordinates corresponds to the plane with z = min_local[2].
        bottom_z = min_z + sphere_radius
        bottom_corners_local = np.array(
            [
                np.array([min_x + sphere_radius, min_y + sphere_radius, bottom_z]),
                np.array([max_x - sphere_radius, min_y + sphere_radius, bottom_z]),
                np.array([min_x + sphere_radius, max_y - sphere_radius, bottom_z]),
                np.array([max_x - sphere_radius, max_y - sphere_radius, bottom_z]),
            ]
        )

        # Transforms back to the STL reference frame.
        bottom_corners_geom = (body_r @ bottom_corners_local.T).T

        # Create a new sphere geom at each transformed corner.
        for idx, corner in enumerate(bottom_corners_geom, start=1):
            # Now directly use corner_world as the correct position for the sphere
            # Make sure we're getting the correct world position with no further offsets.
            sphere_geom = ET.Element("geom")
            sphere_geom.attrib["name"] = f"{mesh_geom_name}_sphere_{idx}"
            sphere_geom.attrib["type"] = "sphere"
            sphere_geom.attrib["pos"] = " ".join(f"{v:.6f}" for v in corner)
            sphere_geom.attrib["size"] = f"{sphere_radius:.6f}"

            # Copies over any other attributes from the original mesh geom.
            for key in ("material", "class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
                if key in mesh_geom.attrib:
                    sphere_geom.attrib[key] = mesh_geom.attrib[key]

            # Add the sphere to the body
            body_elem.append(sphere_geom)

        # Also add a bounding box geom.
        box_size = np.array(
            [
                (max_x - min_x) / 2 - sphere_radius,
                (max_z - min_z) / 2,
                (max_y - min_y) / 2 - sphere_radius,
            ]
        )
        box_pos = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2])
        box_pos = (body_r @ box_pos.T).T
        box_geom = ET.Element("geom")
        box_geom.attrib["name"] = f"{mesh_geom_name}_box"
        box_geom.attrib["type"] = "box"
        box_geom.attrib["pos"] = " ".join(f"{v:.6f}" for v in box_pos)
        box_geom.attrib["size"] = " ".join(f"{v:.6f}" for v in box_size)

        # Copies over any other attributes from the original mesh geom.
        for key in ("material", "class", "condim", "solref", "solimp", "fluidshape", "fluidcoef", "margin"):
            if key in mesh_geom.attrib:
                box_geom.attrib[key] = mesh_geom.attrib[key]

        body_elem.append(box_geom)

        # Remove the original mesh geom from the body.
        body_elem.remove(mesh_geom)
        # logger.info("Removed original mesh geom from link %s", body_name)

    # Save the modified MJCF file.
    save_xml(mjcf_path, tree)
    logger.info("Saved modified MJCF file with feet converted to spheres at %s", mjcf_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MJCF foot mesh geometries into spheres at the bottom corners of their bounding boxes."
    )
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--radius", type=float, required=True, help="Radius of the spheres to create.")
    parser.add_argument("--links", nargs="+", required=True, help="List of link names to convert into foot spheres.")
    args = parser.parse_args()

    convert_feet_to_spheres(args.mjcf_path, args.links, args.radius)


if __name__ == "__main__":
    main()
