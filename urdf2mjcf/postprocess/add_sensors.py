"""Defines a post-processing function that adds sensors to the MJCF model."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from urdf2mjcf.model import ImuSensor

logger = logging.getLogger(__name__)


def add_sensors(
    mjcf_path: str | Path,
    root_link_name: str,
    imus: Sequence[ImuSensor] | None = None,
) -> None:
    """Add sensors to the MJCF model.

    Args:
        mjcf_path: Path to the MJCF file
        root_link_name: Name of the root link
        imus: List of IMU sensor configurations
    """
    tree = ET.parse(mjcf_path)
    mjcf_root = tree.getroot()

    sensor_elem = mjcf_root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(mjcf_root, "sensor")

    def add_base_sensors(link_name: str) -> None:
        ET.SubElement(
            sensor_elem,
            "framepos",
            attrib={
                "name": "base_link_pos",
                "objtype": "site",
                "objname": link_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "framequat",
            attrib={
                "name": "base_link_quat",
                "objtype": "site",
                "objname": link_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "framelinvel",
            attrib={
                "name": "base_link_vel",
                "objtype": "site",
                "objname": link_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "frameangvel",
            attrib={
                "name": "base_link_ang_vel",
                "objtype": "site",
                "objname": link_name,
            },
        )

    if imus:
        for imu in imus:
            # Find the link to attach the IMU to
            link_body = mjcf_root.find(f".//body[@name='{imu.link_name}']")
            if link_body is None:
                logger.warning(f"Link {imu.link_name} not found for IMU sensor")
                continue

            # Create a site for the IMU
            site_name = f"{imu.link_name}_site"

            if len(imu.pos) != 3:
                raise ValueError(f"IMU position must be a 3-element list, got {imu.pos}")
            imu_pos = " ".join(str(p) for p in imu.pos)

            if len(imu.quat) != 4:
                raise ValueError(f"IMU quaternion must be a 4-element list, got {imu.quat}")
            imu_quat = " ".join(str(q) for q in imu.quat)

            # Only make this element if the site doesn't already exist
            site_elem = link_body.find(f".//site[@name='{site_name}']")
            if site_elem is None:
                ET.SubElement(
                    link_body,
                    "site",
                    attrib={
                        "name": site_name,
                        "pos": imu_pos,
                        "quat": imu_quat,
                        "size": "0.01",
                    },
                )

            # Add the accelerometer
            acc_attrib = {
                "name": f"{imu.link_name}_acc",
                "site": site_name,
            }
            if imu.acc_noise is not None:
                acc_attrib["noise"] = str(imu.acc_noise)
            ET.SubElement(sensor_elem, "accelerometer", attrib=acc_attrib)

            # Add the gyroscope
            gyro_attrib = {
                "name": f"{imu.link_name}_gyro",
                "site": site_name,
            }
            if imu.gyro_noise is not None:
                gyro_attrib["noise"] = str(imu.gyro_noise)
            ET.SubElement(sensor_elem, "gyro", attrib=gyro_attrib)

            # Add the magnetometer
            mag_attrib = {
                "name": f"{imu.link_name}_mag",
                "site": site_name,
            }
            if imu.mag_noise is not None:
                mag_attrib["noise"] = str(imu.mag_noise)
            ET.SubElement(sensor_elem, "magnetometer", attrib=mag_attrib)

            # Add other sensors
            add_base_sensors(imu.link_name)
    else:
        add_base_sensors(root_link_name)

    # Save changes
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    add_sensors(args.mjcf_path, "base_link")


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.add_sensors
    main()
