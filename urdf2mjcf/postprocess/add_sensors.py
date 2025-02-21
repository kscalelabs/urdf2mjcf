"""Defines a post-processing function that adds sensors to the MJCF model."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from scipy.spatial.transform import Rotation as R

from urdf2mjcf.model import ConversionMetadata
from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def add_sensors(
    mjcf_path: str | Path,
    root_body_name: str,
    metadata: ConversionMetadata | None = None,
) -> None:
    """Add sensors to the MJCF model.

    Args:
        mjcf_path: Path to the MJCF file
        root_body_name: Name of the root body
        metadata: Metadata for the MJCF model
    """
    if metadata is None:
        metadata = ConversionMetadata()

    tree = ET.parse(mjcf_path)
    mjcf_root = tree.getroot()

    sensor_elem = mjcf_root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(mjcf_root, "sensor")

    def add_base_sensors(site_name: str) -> None:
        ET.SubElement(
            sensor_elem,
            "framepos",
            attrib={
                "name": "base_link_pos",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "framequat",
            attrib={
                "name": "base_link_quat",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "framelinvel",
            attrib={
                "name": "base_link_vel",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "frameangvel",
            attrib={
                "name": "base_link_ang_vel",
                "objtype": "site",
                "objname": site_name,
            },
        )

    if metadata.imus:
        for imu in metadata.imus:
            # Find the link to attach the IMU to
            link_body = mjcf_root.find(f".//body[@name='{imu.body_name}']")
            if link_body is None:
                options = [body.attrib["name"] for body in mjcf_root.findall(".//body")]
                raise ValueError(f"Body {imu.body_name} not found for IMU sensor. Options: {options}")

            # Find the site associated with the link.
            site_elem = link_body.find(".//site")
            if site_elem is None:
                site_elem = ET.SubElement(link_body, "site", name=f"{imu.body_name}_site")
            site_name = site_elem.attrib["name"]

            # Updates the site position and rotation.
            if imu.rpy is not None:
                rotation = R.from_euler("xyz", imu.rpy, degrees=True)
                qx, qy, qz, qw = rotation.as_quat(scalar_first=False)
                site_elem.attrib["quat"] = f"{qw} {qx} {qy} {qz}"

            if imu.pos is not None:
                site_elem.attrib["pos"] = " ".join(str(x) for x in imu.pos)

            # Add the accelerometer
            acc_attrib = {
                "name": f"{imu.body_name}_acc",
                "site": site_name,
            }
            if imu.acc_noise is not None:
                acc_attrib["noise"] = str(imu.acc_noise)
            ET.SubElement(sensor_elem, "accelerometer", attrib=acc_attrib)

            # Add the gyroscope
            gyro_attrib = {
                "name": f"{imu.body_name}_gyro",
                "site": site_name,
            }
            if imu.gyro_noise is not None:
                gyro_attrib["noise"] = str(imu.gyro_noise)
            ET.SubElement(sensor_elem, "gyro", attrib=gyro_attrib)

            # Add the magnetometer
            mag_attrib = {
                "name": f"{imu.body_name}_mag",
                "site": site_name,
            }
            if imu.mag_noise is not None:
                mag_attrib["noise"] = str(imu.mag_noise)
            ET.SubElement(sensor_elem, "magnetometer", attrib=mag_attrib)

            # Add other sensors
            add_base_sensors(site_name)

    else:
        # Finds the root body.
        root_body = mjcf_root.find(f".//body[@name='{root_body_name}']")
        if root_body is None:
            raise ValueError(f"Root body {root_body_name} not found in the MJCF model.")

        # Find the site associated with the root body.
        site_elem = root_body.find(".//site")
        if site_elem is None:
            site_elem = ET.SubElement(root_body, "site", name=f"{root_body_name}_site")
        site_name = site_elem.attrib["name"]

        add_base_sensors(site_name)

    # Find the first <body> element to attach the default cameras instead of the root element.
    first_body = mjcf_root.find(".//body")
    if first_body is None:
        raise ValueError("No <body> element found in the MJCF model to attach cameras.")

    for cam in metadata.cameras:
        attrib = {
            "name": cam.name,
            "mode": cam.mode,
            "fovy": str(cam.fovy),
        }

        if cam.rpy is not None:
            rotation = R.from_euler("xyz", cam.rpy, degrees=True)
            qx, qy, qz, qw = rotation.as_quat(scalar_first=False)
            attrib["quat"] = f"{qw} {qx} {qy} {qz}"

        if cam.pos is not None:
            attrib["pos"] = " ".join(str(x) for x in cam.pos)

        ET.SubElement(first_body, "camera", attrib=attrib)

    # Save changes
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    add_sensors(args.mjcf_path, "base_link")


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.add_sensors
    main()
