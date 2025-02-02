"""Converts URDF files to MJCF files."""

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pydantic import BaseModel

from urdf2mjcf.utils import save_xml


class JointParam(BaseModel):
    kp: float = 0.0
    kd: float = 0.0

    class Config:
        extra = "forbid"


class JointParamsMetadata(BaseModel):
    pd_params: dict[str, JointParam] = {}

    class Config:
        extra = "forbid"


@dataclass
class ParsedJointParams:
    name: str
    type: str
    stiffness: float
    damping: float
    lower: float
    upper: float


# -----------------------------
# Helper functions for transforms
# -----------------------------
def parse_vector(s: str):
    """Convert a string 'x y z' to a list of floats."""
    return list(map(float, s.split()))


def quat_from_str(s: str):
    """Convert a string 'w x y z' to a list of floats."""
    return list(map(float, s.split()))


def quat_to_rot(q):
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]


def build_transform(pos_str: str, quat_str: str):
    """Build a 4x4 homogeneous transformation matrix from pos and quat strings."""
    pos = parse_vector(pos_str)
    q = quat_from_str(quat_str)
    R = quat_to_rot(q)
    T = [
        [R[0][0], R[0][1], R[0][2], pos[0]],
        [R[1][0], R[1][1], R[1][2], pos[1]],
        [R[2][0], R[2][1], R[2][2], pos[2]],
        [0, 0, 0, 1],
    ]
    return T


def mat_mult(A, B):
    """Multiply two 4x4 matrices A and B."""
    result = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(4))
    return result


# -----------------------------
# Constant to adjust the computed offset for foot meshes.
# If a collision/visual geom is of type 'mesh' and the parent body's name contains "foot",
# we subtract this extra offset so that the bottom of the foot aligns with the ground.
# -----------------------------
DEFAULT_FOOT_MESH_BOTTOM_OFFSET = 0.05


# -----------------------------
# Recursive computation of minimum z.
# -----------------------------
def compute_min_z(body: ET.Element, parent_transform) -> float:
    """Recursively compute the minimum z value (in world coordinates) of all geoms
    in the MJCF body subtree. This function uses the parent's transform to compute
    the current body's world transform.
    """
    pos_str = body.attrib.get("pos", "0 0 0")
    quat_str = body.attrib.get("quat", "1 0 0 0")
    T_body = mat_mult(parent_transform, build_transform(pos_str, quat_str))
    local_min_z = float("inf")

    # Look for geoms attached to this body.
    for child in body:
        if child.tag == "geom":
            gpos_str = child.attrib.get("pos", "0 0 0")
            gquat_str = child.attrib.get("quat", "1 0 0 0")
            T_geom = build_transform(gpos_str, gquat_str)
            T_total = mat_mult(T_body, T_geom)
            # The translation part of T_total is in column 3.
            z = T_total[2][3]
            geom_type = child.attrib.get("type", "")
            if geom_type == "box":
                size_vals = list(map(float, child.attrib.get("size", "0 0 0").split()))
                half_height = size_vals[2] if len(size_vals) >= 3 else 0
                candidate = z - half_height
            elif geom_type == "cylinder":
                size_vals = list(map(float, child.attrib.get("size", "0 0").split()))
                half_length = size_vals[1] if len(size_vals) >= 2 else 0
                candidate = z - half_length
            elif geom_type == "sphere":
                r = float(child.attrib.get("size", "0"))
                candidate = z - r
            elif geom_type == "mesh":
                body_name = body.attrib.get("name", "").lower()
                # For foot bodies, subtract an extra offset so that the bottom (not the top) of the foot touches the ground.
                extra = DEFAULT_FOOT_MESH_BOTTOM_OFFSET if "foot" in body_name else 0.0
                candidate = z - extra
            else:
                candidate = z

            local_min_z = min(candidate, local_min_z)

        # Recurse into child bodies.
        elif child.tag == "body":
            child_min = compute_min_z(child, T_body)
            local_min_z = min(child_min, local_min_z)

    return local_min_z


def add_compiler(root: ET.Element) -> None:
    element = ET.Element(
        "compiler",
        attrib={
            "angle": "radian",
            "meshdir": "meshes",
            "eulerseq": "zyx",
            "autolimits": "true",
        },
    )

    if isinstance(existing_element := root.find("compiler"), ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_default(root: ET.Element) -> None:
    default = ET.Element("default")

    # Adds default joint options.
    ET.SubElement(
        default,
        "joint",
        attrib={
            "limited": "true",
            "damping": "0.01",
            "armature": "0.01",
            "frictionloss": "0.01",
        },
    )

    # Adds default geom options.
    ET.SubElement(
        default,
        "geom",
        attrib={
            "condim": "4",
            "contype": "1",
            "conaffinity": "15",
            "friction": "0.9 0.2 0.2",
            "solref": "0.001 2",
        },
    )

    # Adds default motor options.
    ET.SubElement(
        default,
        "motor",
        attrib={"ctrllimited": "true"},
    )

    # Adds default equality options.
    ET.SubElement(
        default,
        "equality",
        attrib={"solref": "0.001 2"},
    )

    # Adds default visualgeom options.
    default_element = ET.SubElement(
        default,
        "default",
        attrib={"class": "visualgeom"},
    )
    ET.SubElement(
        default_element,
        "geom",
        attrib={"material": "visualgeom", "condim": "1", "contype": "0", "conaffinity": "0"},
    )

    if isinstance(existing_element := root.find("default"), ET.Element):
        root.remove(existing_element)
    root.insert(0, default)


def add_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add textures and materials
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "name": "texplane",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".0 .0 .0",
            "rgb2": ".8 .8 .8",
            "width": "100",
            "height": "100",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "matplane",
            "reflectance": "0.",
            "texture": "texplane",
            "texrepeat": "1 1",
            "texuniform": "true",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "visualgeom",
            "rgba": "0.5 0.9 0.2 1",
        },
    )


def add_worldbody_elements(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Add ground plane with a smaller area.
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            attrib={
                "name": "ground",
                "type": "plane",
                "pos": "0 0 0",
                "size": "10 10 0.001",
                "quat": "1 0 0 0",
                "material": "matplane",
                "condim": "3",
                "conaffinity": "15",
            },
        ),
    )

    # Add lights
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.6 0.6 0.6",
                "specular": "0.2 0.2 0.2",
                "pos": "0 0 4",
                "dir": "0 0 -1",
            },
        ),
    )
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.4 0.4 0.4",
                "specular": "0.1 0.1 0.1",
                "pos": "0 0 5.0",
                "dir": "0 0 -1",
                "castshadow": "false",
            },
        ),
    )


def rpy_to_quat(rpy_str: str) -> str:
    """Convert roll, pitch, yaw angles (in radians) to a quaternion (w, x, y, z)."""
    try:
        r, p, y = map(float, rpy_str.split())
    except Exception:
        r, p, y = 0.0, 0.0, 0.0
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return f"{qw} {qx} {qy} {qz}"


def convert_urdf_to_mjcf(
    urdf_path: Union[str, Path],
    mjcf_path: Union[str, Path, None] = None,
    copy_meshes: bool = False,
    joint_params_metadata: JointParamsMetadata | None = None,
) -> None:
    urdf_path = Path(urdf_path)
    mjcf_path = Path(mjcf_path) if mjcf_path is not None else urdf_path.with_suffix(".xml")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    if joint_params_metadata is None:
        joint_params_metadata = JointParamsMetadata()

    # Parse the URDF file.
    urdf_tree = ET.parse(urdf_path)
    robot = urdf_tree.getroot()

    # Create a new MJCF tree root element.
    mjcf_root = ET.Element("mujoco", attrib={"model": robot.attrib.get("name", "converted_robot")})

    # Add compiler, assets, and default settings.
    add_compiler(mjcf_root)
    add_assets(mjcf_root)
    add_default(mjcf_root)

    # Create or find the worldbody element.
    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(mjcf_root, "worldbody")

    # Build mappings for URDF links and joints.
    link_map = {link.attrib["name"]: link for link in robot.findall("link")}
    parent_map = {}
    child_joints = {}
    for joint in robot.findall("joint"):
        parent_name = joint.find("parent").attrib["link"]
        child_name = joint.find("child").attrib["link"]
        parent_map.setdefault(parent_name, []).append((child_name, joint))
        child_joints[child_name] = joint

    all_links = set(link_map.keys())
    child_links = set(child_joints.keys())
    root_links = list(all_links - child_links)
    if not root_links:
        raise ValueError("No root link found in URDF.")
    root_link_name = root_links[0]

    # These are used to collect the mesh assets and actuator joints.
    mesh_assets: dict[str, str] = {}
    actuator_joints: list[ParsedJointParams] = []

    def build_body(
        link_name: str,
        joint: ET.Element = None,
        actuator_joints: list[ParsedJointParams] = None,
    ) -> ET.Element:
        # Retrieve the URDF link.
        link = link_map[link_name]

        if joint is not None:
            origin_elem = joint.find("origin")
            if origin_elem is not None:
                pos = origin_elem.attrib.get("xyz", "0 0 0")
                rpy = origin_elem.attrib.get("rpy", "0 0 0")
                quat = rpy_to_quat(rpy)
            else:
                pos = "0 0 0"
                quat = "1 0 0 0"
        else:
            pos = "0 0 0"
            quat = "1 0 0 0"

        body = ET.Element("body", attrib={"name": link_name, "pos": pos, "quat": quat})

        # Add joint element if this is not the root and the joint is not fixed.
        if joint is not None:
            jtype = joint.attrib.get("type", "fixed")
            if jtype != "fixed":
                j_name = joint.attrib.get("name", link_name + "_joint")
                j_attrib = {"name": j_name}
                if jtype in ["revolute", "continuous"]:
                    j_attrib["type"] = "hinge"
                elif jtype == "prismatic":
                    j_attrib["type"] = "slide"
                else:
                    j_attrib["type"] = jtype
                limit = joint.find("limit")
                if limit is not None:
                    lower = limit.attrib.get("lower")
                    upper = limit.attrib.get("upper")
                    if lower is not None and upper is not None:
                        j_attrib["range"] = f"{lower} {upper}"
                else:
                    lower = None
                    upper = None
                axis_elem = joint.find("axis")
                if axis_elem is not None:
                    j_attrib["axis"] = axis_elem.attrib.get("xyz", "0 0 1")
                ET.SubElement(body, "joint", attrib=j_attrib)

                # Use PD gains from the suffix mapping dictionary instead of URDF dynamics.
                for suffix, param in joint_params_metadata.pd_params.items():
                    if j_name.endswith(suffix):
                        stiffness_val = param.kp
                        damping_val = param.kd
                        break
                else:
                    stiffness_val = 0.0
                    damping_val = 0.0

                if actuator_joints is not None:
                    actuator_joints.append(
                        ParsedJointParams(
                            name=j_name,
                            type=j_attrib["type"],
                            stiffness=stiffness_val,
                            damping=damping_val,
                            lower=lower,
                            upper=upper,
                        )
                    )

        # Process inertial information.
        inertial = link.find("inertial")
        if inertial is not None:
            inertial_elem = ET.Element("inertial")
            origin_inertial = inertial.find("origin")
            if origin_inertial is not None:
                inertial_elem.attrib["pos"] = origin_inertial.attrib.get("xyz", "0 0 0")
                rpy = origin_inertial.attrib.get("rpy", "0 0 0")
                inertial_elem.attrib["quat"] = rpy_to_quat(rpy)
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                inertial_elem.attrib["mass"] = mass_elem.attrib.get("value", "0")
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.attrib.get("ixx", "0"))
                ixy = float(inertia_elem.attrib.get("ixy", "0"))
                ixz = float(inertia_elem.attrib.get("ixz", "0"))
                iyy = float(inertia_elem.attrib.get("iyy", "0"))
                iyz = float(inertia_elem.attrib.get("iyz", "0"))
                izz = float(inertia_elem.attrib.get("izz", "0"))
                if abs(ixy) > 1e-6 or abs(ixz) > 1e-6 or abs(iyz) > 1e-6:
                    print(
                        f"Warning: off-diagonal inertia terms for link '{link_name}' are nonzero and will be ignored."
                    )
                inertial_elem.attrib["diaginertia"] = f"{ixx} {iyy} {izz}"
            body.append(inertial_elem)

        # Process collision geometries.
        collisions = link.findall("collision")
        for idx, collision in enumerate(collisions):
            origin_collision = collision.find("origin")
            if origin_collision is not None:
                pos_geom = origin_collision.attrib.get("xyz", "0 0 0")
                rpy_geom = origin_collision.attrib.get("rpy", "0 0 0")
                quat_geom = rpy_to_quat(rpy_geom)
            else:
                pos_geom = "0 0 0"
                quat_geom = "1 0 0 0"
            geom_attrib = {"name": f"{link_name}_collision_{idx}", "pos": pos_geom, "quat": quat_geom}
            geom_elem = collision.find("geometry")
            if geom_elem is not None:
                if geom_elem.find("box") is not None:
                    box_elem = geom_elem.find("box")
                    size_str = box_elem.attrib.get("size", "1 1 1")
                    half_sizes = " ".join(str(float(s) / 2) for s in size_str.split())
                    geom_attrib["type"] = "box"
                    geom_attrib["size"] = half_sizes
                elif geom_elem.find("cylinder") is not None:
                    cyl_elem = geom_elem.find("cylinder")
                    radius = cyl_elem.attrib.get("radius", "0.1")
                    length = cyl_elem.attrib.get("length", "1")
                    geom_attrib["type"] = "cylinder"
                    geom_attrib["size"] = f"{radius} {float(length)/2}"
                elif geom_elem.find("sphere") is not None:
                    sph_elem = geom_elem.find("sphere")
                    radius = sph_elem.attrib.get("radius", "0.1")
                    geom_attrib["type"] = "sphere"
                    geom_attrib["size"] = radius
                elif geom_elem.find("mesh") is not None:
                    mesh_elem = geom_elem.find("mesh")
                    filename = mesh_elem.attrib.get("filename")
                    if filename is not None:
                        mesh_name = Path(filename).name
                        if mesh_name not in mesh_assets:
                            mesh_assets[mesh_name] = filename
                        geom_attrib["type"] = "mesh"
                        geom_attrib["mesh"] = mesh_name
                        if "scale" in mesh_elem.attrib:
                            geom_attrib["scale"] = mesh_elem.attrib["scale"]
            geom_attrib["rgba"] = "0 0 0 0"
            ET.SubElement(body, "geom", attrib=geom_attrib)

        # Process visual geometries.
        visuals = link.findall("visual")
        for idx, visual in enumerate(visuals):
            origin_visual = visual.find("origin")
            if origin_visual is not None:
                pos_geom = origin_visual.attrib.get("xyz", "0 0 0")
                rpy_geom = origin_visual.attrib.get("rpy", "0 0 0")
                quat_geom = rpy_to_quat(rpy_geom)
            else:
                pos_geom = "0 0 0"
                quat_geom = "1 0 0 0"
            geom_attrib = {
                "name": f"{link_name}_visual_{idx}",
                "pos": pos_geom,
                "quat": quat_geom,
                "material": "visualgeom",
                "contype": "0",
                "conaffinity": "0",
            }
            geom_elem = visual.find("geometry")
            if geom_elem is not None:
                if geom_elem.find("box") is not None:
                    box_elem = geom_elem.find("box")
                    size_str = box_elem.attrib.get("size", "1 1 1")
                    half_sizes = " ".join(str(float(s) / 2) for s in size_str.split())
                    geom_attrib["type"] = "box"
                    geom_attrib["size"] = half_sizes
                elif geom_elem.find("cylinder") is not None:
                    cyl_elem = geom_elem.find("cylinder")
                    radius = cyl_elem.attrib.get("radius", "0.1")
                    length = cyl_elem.attrib.get("length", "1")
                    geom_attrib["type"] = "cylinder"
                    geom_attrib["size"] = f"{radius} {float(length)/2}"
                elif geom_elem.find("sphere") is not None:
                    sph_elem = geom_elem.find("sphere")
                    radius = sph_elem.attrib.get("radius", "0.1")
                    geom_attrib["type"] = "sphere"
                    geom_attrib["size"] = radius
                elif geom_elem.find("mesh") is not None:
                    mesh_elem = geom_elem.find("mesh")
                    filename = mesh_elem.attrib.get("filename")
                    if filename is not None:
                        mesh_name = Path(filename).name
                        if mesh_name not in mesh_assets:
                            mesh_assets[mesh_name] = filename
                        geom_attrib["type"] = "mesh"
                        geom_attrib["mesh"] = mesh_name
                        if "scale" in mesh_elem.attrib:
                            geom_attrib["scale"] = mesh_elem.attrib["scale"]
            ET.SubElement(body, "geom", attrib=geom_attrib)

        # Recurse for child links.
        if link_name in parent_map:
            for child_name, child_joint in parent_map[link_name]:
                child_body = build_body(child_name, child_joint, actuator_joints)
                body.append(child_body)
        return body

    # Build the robot body hierarchy starting from the root link.
    robot_body = build_body(root_link_name, None, actuator_joints)

    # Automatically compute the base offset from the model's minimum z coordinate.
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    min_z = compute_min_z(robot_body, identity)
    computed_offset = -min_z
    print(f"Auto-detected base offset: {computed_offset} (min z = {min_z})")

    # Create a root body with a freejoint and an IMU site; the root body's z position uses the computed offset.
    root_body = ET.Element("body", attrib={"name": "root", "pos": f"0 0 {computed_offset}", "quat": "1 0 0 0"})
    ET.SubElement(root_body, "freejoint", attrib={"name": "root"})
    ET.SubElement(root_body, "site", attrib={"name": "imu", "size": "0.01", "pos": "0 0 0"})
    root_body.append(robot_body)
    worldbody.append(root_body)

    # Replace the previous actuator block with one that uses positional control.
    actuator_elem = ET.SubElement(mjcf_root, "actuator")
    for joint in actuator_joints:
        ctrlrange = f"{joint.lower} {joint.upper}" if joint.lower is not None and joint.upper is not None else "-1 1"
        ET.SubElement(
            actuator_elem,
            "position",
            attrib={
                "name": f"{joint.name}_ctrl",
                "joint": joint.name,
                "kp": f"{joint.stiffness:.8f}",
                "kv": f"{joint.damping:.8f}",
                "ctrlrange": ctrlrange,
            },
        )

    # Add additional worldbody elements (ground, lights, etc.).
    add_worldbody_elements(mjcf_root)

    # Add mesh assets to the asset section.
    asset_elem = mjcf_root.find("asset")
    for mesh_name, filename in mesh_assets.items():
        ET.SubElement(asset_elem, "mesh", attrib={"name": mesh_name, "file": Path(filename).name})

    # Copy mesh files if requested.
    if copy_meshes:
        urdf_dir = urdf_path.parent.resolve()
        target_mesh_dir = (mjcf_path.parent / "meshes").resolve()
        target_mesh_dir.mkdir(parents=True, exist_ok=True)
        for mesh_name, filename in mesh_assets.items():
            source_path = (urdf_dir / filename).resolve()
            target_path = target_mesh_dir / Path(filename).name
            if source_path != target_path:
                shutil.copy2(source_path, target_path)

    # Save the updated MJCF file.
    save_xml(mjcf_path, ET.ElementTree(mjcf_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a URDF file to an MJCF file.")

    parser.add_argument(
        "urdf_path",
        type=str,
        help="The path to the URDF file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The path to the output MJCF file.",
    )
    parser.add_argument(
        "--copy-meshes",
        action="store_true",
        help="Copy mesh files to the output MJCF directory.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="A JSON string containing joint PD parameters.",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        help="A JSON file containing joint PD parameters.",
    )
    args = parser.parse_args()

    # Parses the raw metadata from the command line arguments.
    raw_metadata: dict | None = None
    if args.metadata_file is not None and args.metadata is not None:
        raise ValueError("Cannot specify both --metadata and --metadata-file")
    elif args.metadata_file is not None:
        with open(args.metadata_file, "r") as f:
            raw_metadata = json.load(f)
    elif args.metadata is not None:
        raw_metadata = json.loads(args.metadata)
    else:
        raw_metadata = None

    metadata = None if raw_metadata is None else JointParamsMetadata.model_validate(raw_metadata, strict=True)

    convert_urdf_to_mjcf(
        urdf_path=args.urdf_path,
        mjcf_path=args.output,
        copy_meshes=args.copy_meshes,
        joint_params_metadata=metadata,
    )


if __name__ == "__main__":
    main()
