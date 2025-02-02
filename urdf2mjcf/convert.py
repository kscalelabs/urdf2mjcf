"""Converts URDF files to MJCF files."""

import argparse
import json
import logging
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import colorlogging
from pydantic import BaseModel

from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


class JointParam(BaseModel):
    kp: float | None = None
    kd: float | None = None

    class Config:
        extra = "forbid"


class JointParamsMetadata(BaseModel):
    suffix_to_pd_params: dict[str, JointParam] = {}
    default: JointParam = JointParam()

    class Config:
        extra = "forbid"


@dataclass
class ParsedJointParams:
    """Parsed joint parameters from URDF.

    Attributes:
        name: Joint name.
        type: Joint type (hinge, slide, etc.).
        stiffness: Joint stiffness (kp).
        damping: Joint damping (kd).
        lower: Lower joint limit, if any.
        upper: Upper joint limit, if any.
        max_torque: Maximum joint torque, if any.
    """

    name: str
    type: str
    stiffness: float | None = None
    damping: float | None = None
    max_torque: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass
class GeomElement:
    type: str
    size: str | None = None
    scale: str | None = None
    mesh: str | None = None


def parse_vector(s: str) -> list[float]:
    """Convert a string of space-separated numbers to a list of floats.

    Args:
        s: Space-separated string of numbers (e.g., "1 2 3").

    Returns:
        List of parsed float values.
    """
    return list(map(float, s.split()))


def quat_from_str(s: str) -> list[float]:
    """Convert a quaternion string to a list of floats.

    Args:
        s: Space-separated string of quaternion values (w x y z).

    Returns:
        List of parsed quaternion values [w, x, y, z].
    """
    return list(map(float, s.split()))


def quat_to_rot(q: list[float]) -> list[list[float]]:
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


def build_transform(pos_str: str, quat_str: str) -> list[list[float]]:
    """Build a 4x4 homogeneous transformation matrix from position and quaternion strings.

    Args:
        pos_str: Space-separated string of position values (x y z).
        quat_str: Space-separated string of quaternion values (w x y z).

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    pos = parse_vector(pos_str)
    q = quat_from_str(quat_str)
    r_mat = quat_to_rot(q)
    transform = [
        [r_mat[0][0], r_mat[0][1], r_mat[0][2], pos[0]],
        [r_mat[1][0], r_mat[1][1], r_mat[1][2], pos[1]],
        [r_mat[2][0], r_mat[2][1], r_mat[2][2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return transform


def mat_mult(mat_a: list[list[float]], mat_b: list[list[float]]) -> list[list[float]]:
    """Multiply two 4x4 matrices A and B."""
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            result[i][j] = sum(mat_a[i][k] * mat_b[k][j] for k in range(4))
    return result


DEFAULT_FOOT_MESH_BOTTOM_OFFSET: float = 0.05


def compute_min_z(body: ET.Element, parent_transform: list[list[float]]) -> float:
    """Recursively computes the minimum Z value in the world frame.

    This is used to compute the starting height of the robot.

    Args:
        body: The current body element.
        parent_transform: The transform of the parent body.

    Returns:
        The minimum Z value in the world frame.
    """
    pos_str: str = body.attrib.get("pos", "0 0 0")
    quat_str: str = body.attrib.get("quat", "1 0 0 0")
    body_tf: list[list[float]] = mat_mult(parent_transform, build_transform(pos_str, quat_str))
    local_min_z: float = float("inf")

    for child in body:
        if child.tag == "geom":
            gpos_str: str = child.attrib.get("pos", "0 0 0")
            gquat_str: str = child.attrib.get("quat", "1 0 0 0")
            geom_tf: list[list[float]] = build_transform(gpos_str, gquat_str)
            total_tf: list[list[float]] = mat_mult(body_tf, geom_tf)

            # The translation part of T_total is in column 3.
            z: float = total_tf[2][3]
            geom_type: str = child.attrib.get("type", "")
            if geom_type == "box":
                size_vals: list[float] = list(map(float, child.attrib.get("size", "0 0 0").split()))
                half_height: float = size_vals[2] if len(size_vals) >= 3 else 0.0
                candidate: float = z - half_height
            elif geom_type == "cylinder":
                size_vals = list(map(float, child.attrib.get("size", "0 0").split()))
                half_length: float = size_vals[1] if len(size_vals) >= 2 else 0.0
                candidate = z - half_length
            elif geom_type == "sphere":
                r = float(child.attrib.get("size", "0"))
                candidate = z - r
            elif geom_type == "mesh":
                body_name: str = body.attrib.get("name", "").lower()
                extra: float = DEFAULT_FOOT_MESH_BOTTOM_OFFSET if "foot" in body_name else 0.0
                candidate = z - extra
            else:
                candidate = z

            local_min_z = min(candidate, local_min_z)

        elif child.tag == "body":
            child_min: float = compute_min_z(child, body_tf)
            local_min_z = min(child_min, local_min_z)

    return local_min_z


def add_compiler(root: ET.Element) -> None:
    """Add a compiler element to the MJCF root.

    Args:
        root: The MJCF root element.
        timestep: Optional control timestep in seconds. If specified, sets the simulation
                 timestep for the PD controller.
    """
    attrib = {
        "angle": "radian",
        "meshdir": "meshes",
        "eulerseq": "zyx",
        "autolimits": "true",
    }

    element = ET.Element("compiler", attrib=attrib)
    existing_element = root.find("compiler")
    if isinstance(existing_element, ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_default(root: ET.Element) -> None:
    """Add default settings for joints, geoms, motors, and equality constraints."""
    default = ET.Element("default")
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
    ET.SubElement(
        default,
        "motor",
        attrib={"ctrllimited": "true"},
    )
    ET.SubElement(
        default,
        "equality",
        attrib={"solref": "0.001 2"},
    )
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
    existing_element = root.find("default")
    if isinstance(existing_element, ET.Element):
        root.remove(existing_element)
    root.insert(0, default)


def add_assets(root: ET.Element, materials: dict[str, str]) -> None:
    """Add texture and material assets to the MJCF root.

    Args:
        root: The MJCF root element.
        materials: Dictionary mapping material names to RGBA color strings.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
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

    # Add materials from URDF
    for name, rgba in materials.items():
        ET.SubElement(
            asset,
            "material",
            attrib={
                "name": name,
                "rgba": rgba,
            },
        )

    # Add default material for visual elements without materials
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "default_material",
            "rgba": "0.7 0.7 0.7 1",
        },
    )


def add_worldbody_elements(root: ET.Element) -> None:
    """Add the ground plane and lights to the worldbody element."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Add ground plane.
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

    # Add lights.
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
    urdf_path: str | Path,
    mjcf_path: str | Path | None = None,
    copy_meshes: bool = False,
    joint_params_metadata: JointParamsMetadata | None = None,
) -> None:
    """Converts a URDF file to an MJCF file.

    Args:
        urdf_path: The path to the URDF file.
        mjcf_path: The desired output MJCF file path. If None, the URDF path
            with an ".xml" extension is used.
        copy_meshes: If True, mesh files will be copied to the MJCF meshes
            directory.
        joint_params_metadata: Optional PD gains metadata for joints.
    """
    urdf_path = Path(urdf_path)
    mjcf_path = Path(mjcf_path) if mjcf_path is not None else urdf_path.with_suffix(".xml")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    if joint_params_metadata is None:
        joint_params_metadata = JointParamsMetadata()

    # Parse the URDF file.
    urdf_tree: ET.ElementTree = ET.parse(urdf_path)
    robot: ET.Element = urdf_tree.getroot()

    # Parse materials from URDF - both from root level and from link visuals
    materials: dict[str, str] = {}

    # Get materials defined at the robot root level
    for material in robot.findall("material"):
        name = material.attrib.get("name")
        if name is None:
            continue
        color = material.find("color")
        if color is not None:
            rgba = color.attrib.get("rgba")
            if rgba is not None:
                materials[name] = rgba

    # Get materials defined in link visual elements
    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            visual_material = visual.find("material")
            if visual_material is None:
                continue
            name = visual_material.attrib.get("name")
            if name is None:
                continue
            color = visual_material.find("color")
            if color is not None:
                rgba = color.attrib.get("rgba")
                if rgba is not None:
                    materials[name] = rgba

    # Create a new MJCF tree root element.
    mjcf_root: ET.Element = ET.Element("mujoco", attrib={"model": robot.attrib.get("name", "converted_robot")})

    # Add compiler, assets, and default settings.
    add_compiler(mjcf_root)
    add_assets(mjcf_root, materials)
    add_default(mjcf_root)

    # Create or find the worldbody element.
    worldbody: ET.Element | None = mjcf_root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(mjcf_root, "worldbody")

    # Build mappings for URDF links and joints.
    link_map: dict[str, ET.Element] = {link.attrib["name"]: link for link in robot.findall("link")}
    parent_map: dict[str, list[tuple[str, ET.Element]]] = {}
    child_joints: dict[str, ET.Element] = {}
    for joint in robot.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            logger.warning("Joint missing parent or child element")
            continue
        parent_name = parent_elem.attrib.get("link", "")
        child_name = child_elem.attrib.get("link", "")
        if not parent_name or not child_name:
            logger.warning("Joint missing parent or child link name")
            continue
        parent_map.setdefault(parent_name, []).append((child_name, joint))
        child_joints[child_name] = joint

    all_links = set(link_map.keys())
    child_links = set(child_joints.keys())
    root_links: list[str] = list(all_links - child_links)
    if not root_links:
        raise ValueError("No root link found in URDF.")
    root_link_name: str = root_links[0]

    # These dictionaries are used to collect mesh assets and actuator joints.
    mesh_assets: dict[str, str] = {}
    actuator_joints: list[ParsedJointParams] = []

    def handle_geom_element(geom_elem: ET.Element | None, default_size: str) -> GeomElement:
        """Helper to handle geometry elements safely.

        Args:
            geom_elem: The geometry element to process
            default_size: Default size to use if not specified

        Returns:
            A GeomElement instance
        """
        if geom_elem is None:
            return GeomElement(type="box", size=default_size, scale=None, mesh=None)

        box_elem = geom_elem.find("box")
        if box_elem is not None:
            size_str = box_elem.attrib.get("size", default_size)
            return GeomElement(
                type="box",
                size=" ".join(str(float(s) / 2) for s in size_str.split()),
            )

        cyl_elem = geom_elem.find("cylinder")
        if cyl_elem is not None:
            radius = cyl_elem.attrib.get("radius", "0.1")
            length = cyl_elem.attrib.get("length", "1")
            return GeomElement(
                type="cylinder",
                size=f"{radius} {float(length) / 2}",
            )

        sph_elem = geom_elem.find("sphere")
        if sph_elem is not None:
            radius = sph_elem.attrib.get("radius", "0.1")
            return GeomElement(
                type="sphere",
                size=radius,
            )

        mesh_elem = geom_elem.find("mesh")
        if mesh_elem is not None:
            filename = mesh_elem.attrib.get("filename")
            if filename is not None:
                mesh_name = Path(filename).name
                if mesh_name not in mesh_assets:
                    mesh_assets[mesh_name] = filename
                scale = mesh_elem.attrib.get("scale")
                return GeomElement(
                    type="mesh",
                    size=None,
                    scale=scale,
                    mesh=mesh_name,
                )

        return GeomElement(
            type="box",
            size=default_size,
        )

    def build_body(
        link_name: str,
        joint: ET.Element | None = None,
        actuator_joints: list[ParsedJointParams] = actuator_joints,
    ) -> ET.Element:
        """Recursively build a MJCF body element from a URDF link."""
        link: ET.Element = link_map[link_name]

        if joint is not None:
            origin_elem: ET.Element | None = joint.find("origin")
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

        body: ET.Element = ET.Element("body", attrib={"name": link_name, "pos": pos, "quat": quat})

        # Add joint element if this is not the root and the joint type is not fixed.
        if joint is not None:
            jtype: str = joint.attrib.get("type", "fixed")
            if jtype != "fixed":
                j_name: str = joint.attrib.get("name", link_name + "_joint")
                j_attrib: dict[str, str] = {"name": j_name}
                if jtype in ["revolute", "continuous"]:
                    j_attrib["type"] = "hinge"
                elif jtype == "prismatic":
                    j_attrib["type"] = "slide"
                else:
                    j_attrib["type"] = jtype
                limit = joint.find("limit")
                if limit is not None:
                    lower_val = limit.attrib.get("lower")
                    upper_val = limit.attrib.get("upper")
                    effort_val = limit.attrib.get("effort")
                    if lower_val is not None and upper_val is not None:
                        j_attrib["range"] = f"{lower_val} {upper_val}"
                        lower_num: float | None = float(lower_val)
                        upper_num: float | None = float(upper_val)
                    else:
                        lower_num = upper_num = None
                    max_torque: float | None = float(effort_val) if effort_val is not None else None
                else:
                    lower_num = upper_num = max_torque = None
                axis_elem = joint.find("axis")
                if axis_elem is not None:
                    j_attrib["axis"] = axis_elem.attrib.get("xyz", "0 0 1")
                ET.SubElement(body, "joint", attrib=j_attrib)

                # Use PD gains from the joint parameters metadata.
                for suffix, param in joint_params_metadata.suffix_to_pd_params.items():
                    if j_name.endswith(suffix):
                        stiffness_val = param.kp
                        damping_val = param.kd
                        break
                else:
                    stiffness_val = joint_params_metadata.default.kp
                    damping_val = joint_params_metadata.default.kd

                actuator_joints.append(
                    ParsedJointParams(
                        name=j_name,
                        type=j_attrib["type"],
                        stiffness=stiffness_val,
                        damping=damping_val,
                        lower=lower_num,
                        upper=upper_num,
                        max_torque=max_torque,
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
                    logger.warning(
                        "Warning: off-diagonal inertia terms for link '%s' are nonzero and will be ignored.",
                        link_name,
                    )
                inertial_elem.attrib["diaginertia"] = f"{ixx} {iyy} {izz}"
            body.append(inertial_elem)

        # Process collision geometries.
        collisions = link.findall("collision")
        for idx, collision in enumerate(collisions):
            origin_collision = collision.find("origin")
            if origin_collision is not None:
                pos_geom: str = origin_collision.attrib.get("xyz", "0 0 0")
                rpy_geom: str = origin_collision.attrib.get("rpy", "0 0 0")
                quat_geom: str = rpy_to_quat(rpy_geom)
            else:
                pos_geom = "0 0 0"
                quat_geom = "1 0 0 0"
            geom_attrib: dict[str, str] = {"name": f"{link_name}_collision_{idx}", "pos": pos_geom, "quat": quat_geom}
            geom_elem: ET.Element | None = collision.find("geometry")
            if geom_elem is not None:
                geom = handle_geom_element(geom_elem, "1 1 1")
                geom_attrib["type"] = geom.type
                if geom.type == "mesh":
                    if geom.mesh is not None:
                        geom_attrib["mesh"] = geom.mesh
                elif geom.size is not None:
                    geom_attrib["size"] = geom.size
                if geom.scale is not None:
                    geom_attrib["scale"] = geom.scale
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
            }

            # Get material from visual element
            material_elem = visual.find("material")
            if material_elem is not None:
                material_name = material_elem.attrib.get("name")
                if material_name in materials:
                    geom_attrib["material"] = material_name
                else:
                    geom_attrib["material"] = "default_material"
            else:
                geom_attrib["material"] = "default_material"

            geom_elem = visual.find("geometry")
            if geom_elem is not None:
                geom = handle_geom_element(geom_elem, "1 1 1")
                geom_attrib["type"] = geom.type
                if geom.type == "mesh":
                    if geom.mesh is not None:
                        geom_attrib["mesh"] = geom.mesh
                elif geom.size is not None:
                    geom_attrib["size"] = geom.size
                if geom.scale is not None:
                    geom_attrib["scale"] = geom.scale
            ET.SubElement(body, "geom", attrib=geom_attrib)

        # Recurse into child links.
        if link_name in parent_map:
            for child_name, child_joint in parent_map[link_name]:
                child_body = build_body(child_name, child_joint, actuator_joints)
                body.append(child_body)
        return body

    # Build the robot body hierarchy starting from the root link.
    robot_body: ET.Element = build_body(root_link_name, None, actuator_joints)

    # Automatically compute the base offset using the model's minimum z coordinate.
    identity: list[list[float]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    min_z: float = compute_min_z(robot_body, identity)
    computed_offset: float = -min_z
    logger.info("Auto-detected base offset: %s (min z = %s)", computed_offset, min_z)

    # Create a root body with a freejoint and an IMU site; the z position uses the computed offset.
    root_body = ET.Element("body", attrib={"name": "root", "pos": f"0 0 {computed_offset}", "quat": "1 0 0 0"})
    ET.SubElement(root_body, "freejoint", attrib={"name": "root"})
    ET.SubElement(root_body, "site", attrib={"name": "imu", "size": "0.01", "pos": "0 0 0"})
    root_body.append(robot_body)
    worldbody.append(root_body)

    # Replace the actuator block with one that uses positional control.
    actuator_elem = ET.SubElement(mjcf_root, "actuator")
    for joint_params in actuator_joints:
        attrib: dict[str, str] = {
            "name": f"{joint_params.name}_ctrl",
            "joint": joint_params.name,
        }

        # Adds the joint PD parameters.
        if joint_params.stiffness is not None:
            attrib["kp"] = f"{joint_params.stiffness:.8f}"
        if joint_params.damping is not None:
            attrib["kv"] = f"{joint_params.damping:.8f}"

        # Adds the joint torque limit.
        if joint_params.max_torque is not None:
            attrib["forcerange"] = f"-{joint_params.max_torque:.8f} {joint_params.max_torque:.8f}"
            attrib["forcelimited"] = "true"

        # Adds the joint limits.
        if joint_params.lower is not None and joint_params.upper is not None:
            attrib["ctrlrange"] = f"{joint_params.lower} {joint_params.upper}"
            attrib["ctrllimited"] = "true"
        elif joint_params.lower is not None or joint_params.upper is not None:
            raise ValueError(f"Joint {joint_params.name} has a lower or upper limit but no ctrlrange")

        ET.SubElement(actuator_elem, "position", attrib=attrib)

    # Add additional worldbody elements (ground, lights, etc.).
    add_worldbody_elements(mjcf_root)

    # Add mesh assets to the asset section.
    asset_elem: ET.Element | None = mjcf_root.find("asset")
    if asset_elem is None:
        asset_elem = ET.SubElement(mjcf_root, "asset")
    for mesh_name, filename in mesh_assets.items():
        ET.SubElement(asset_elem, "mesh", attrib={"name": mesh_name, "file": Path(filename).name})

    # Copy mesh files if requested.
    if copy_meshes:
        urdf_dir: Path = urdf_path.parent.resolve()
        target_mesh_dir: Path = (mjcf_path.parent / "meshes").resolve()
        target_mesh_dir.mkdir(parents=True, exist_ok=True)
        for mesh_name, filename in mesh_assets.items():
            source_path: Path = (urdf_dir / filename).resolve()
            target_path: Path = target_mesh_dir / Path(filename).name
            if source_path != target_path:
                shutil.copy2(source_path, target_path)

    # Save the updated MJCF file.
    save_xml(mjcf_path, ET.ElementTree(mjcf_root))


def main() -> None:
    """Parse command-line arguments and execute the URDF to MJCF conversion."""
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

    colorlogging.configure()

    # Parse the raw metadata from the command line arguments.
    raw_metadata: dict | None = None
    if args.metadata_file is not None and args.metadata is not None:
        raise ValueError("Cannot specify both --metadata and --metadata-file")
    elif args.metadata_file is not None:
        with open(args.metadata_file, "r") as f:
            raw_metadata = json.load(f)
    elif args.metadata is not None:
        raw_metadata = json.loads(args.metadata)
    metadata: JointParamsMetadata | None = (
        None if raw_metadata is None else JointParamsMetadata.model_validate(raw_metadata, strict=True)
    )

    convert_urdf_to_mjcf(
        urdf_path=args.urdf_path,
        mjcf_path=args.output,
        copy_meshes=args.copy_meshes,
        joint_params_metadata=metadata,
    )


if __name__ == "__main__":
    main()
