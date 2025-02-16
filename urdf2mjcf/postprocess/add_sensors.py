"""Defines a post-processing function that adds sensors to the MJCF model."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.model import ConversionMetadata
from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def add_sensors(
    mjcf_path: str | Path,
    root_site_name: str,
    metadata: ConversionMetadata | None = None,
) -> None:
    """Add sensors to the MJCF model.

    Args:
        mjcf_path: Path to the MJCF file
        root_site_name: Name of the root site
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
            link_site = mjcf_root.find(f".//site[@name='{imu.site_name}']")
            if link_site is None:
                options = [site.attrib["name"] for site in mjcf_root.findall(".//site")]
                raise ValueError(f"Site {imu.site_name} not found for IMU sensor. Options: {options}")

            # Add the accelerometer
            acc_attrib = {
                "name": f"{imu.site_name}_acc",
                "site": imu.site_name,
            }
            if imu.acc_noise is not None:
                acc_attrib["noise"] = str(imu.acc_noise)
            ET.SubElement(sensor_elem, "accelerometer", attrib=acc_attrib)

            # Add the gyroscope
            gyro_attrib = {
                "name": f"{imu.site_name}_gyro",
                "site": imu.site_name,
            }
            if imu.gyro_noise is not None:
                gyro_attrib["noise"] = str(imu.gyro_noise)
            ET.SubElement(sensor_elem, "gyro", attrib=gyro_attrib)

            # Add the magnetometer
            mag_attrib = {
                "name": f"{imu.site_name}_mag",
                "site": imu.site_name,
            }
            if imu.mag_noise is not None:
                mag_attrib["noise"] = str(imu.mag_noise)
            ET.SubElement(sensor_elem, "magnetometer", attrib=mag_attrib)

            # Add other sensors
            add_base_sensors(imu.site_name)

    else:
        add_base_sensors(root_site_name)

    # Find the first <body> element to attach the default cameras instead of the root element.
    first_body = mjcf_root.find(".//body")
    if first_body is None:
        raise ValueError("No <body> element found in the MJCF model to attach cameras.")

    for cam in metadata.cameras:
        ET.SubElement(
            first_body,
            "camera",
            attrib={
                "name": cam.name,
                "mode": cam.mode,
                "pos": " ".join(str(x) for x in cam.pos),
                "quat": " ".join(str(x) for x in cam.quat),
                "fovy": str(cam.fovy),
            },
        )

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
