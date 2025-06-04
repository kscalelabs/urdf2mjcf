"""Defines a post-processing function that adds sensors to the MJCF model."""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from scipy.spatial.transform import Rotation as R

from urdf2mjcf.model import ConversionMetadata, SiteMetadata
from urdf2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


class BodyNotFoundError(ValueError):
    """Exception raised when a body is not found in the MJCF model."""

    def __init__(self, body_name: str, available_bodies: list[str]) -> None:
        self.body_name = body_name
        self.available_bodies = available_bodies
        super().__init__(f"Body '{body_name}' not found in the MJCF model. Available bodies: {available_bodies}")


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

    def add_sensors(site_name: str) -> None:
        ET.SubElement(
            sensor_elem,
            "framepos",
            attrib={
                "name": f"{site_name}_pos",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "framequat",
            attrib={
                "name": f"{site_name}_quat",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "framelinvel",
            attrib={
                "name": f"{site_name}_linvel",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "frameangvel",
            attrib={
                "name": f"{site_name}_angvel",
                "objtype": "site",
                "objname": site_name,
            },
        )
        ET.SubElement(
            sensor_elem,
            "velocimeter",
            attrib={
                "name": f"{site_name}_vel",
                "site": site_name,
            },
        )

    def find_site(body_name: str, site_name: str, site_metadata: list[SiteMetadata]) -> ET.Element:
        # Find the body to attach the site to
        link_body = mjcf_root.find(f".//body[@name='{body_name}']")
        if link_body is None:
            options = [body.attrib["name"] for body in mjcf_root.findall(".//body")]
            raise BodyNotFoundError(body_name, options)

        # Find or create the site within the body
        site_elem = link_body.find(f"./site[@name='{site_name}']")
        if site_elem is None:
            site_meta = next(
                (s for s in site_metadata if s.name == site_name), SiteMetadata(body_name=body_name, name=site_name)
            )

            site_elem = ET.SubElement(link_body, "site", name=site_name)
            if site_meta.size is not None:
                site_elem.attrib["size"] = " ".join(str(x) for x in site_meta.size)
            if site_meta.pos is not None:
                site_elem.attrib["pos"] = " ".join(str(x) for x in site_meta.pos)
            if site_meta.site_type is not None:
                site_elem.attrib["type"] = site_meta.site_type
            logger.info(f"Created site '{site_name}' on body '{body_name}'.")

        return site_elem

    # Finds the root body.
    root_body = mjcf_root.find(f".//body[@name='{root_body_name}']")
    if root_body is None:
        raise ValueError(f"Root body {root_body_name} not found in the MJCF model.")

    # Find the site associated with the root body.
    site_elem = root_body.find(".//site")
    if site_elem is None:
        site_elem = ET.SubElement(root_body, "site", name=f"{root_body_name}_site")
    site_name = site_elem.attrib["name"]

    add_sensors(site_name)

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
            add_sensors(site_name)

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

    # Add force sensors.
    for fs in metadata.force_sensors:
        site_elem = find_site(body_name=fs.body_name, site_name=fs.site_name, site_metadata=metadata.sites)

        # Add the force sensor element
        fs_name = fs.name if fs.name else f"{fs.site_name}_force"
        fs_attrib = {
            "name": fs_name,
            "site": fs.site_name,  # Use the site_name directly
        }
        if fs.noise is not None:
            fs_attrib["noise"] = str(fs.noise)

        ET.SubElement(sensor_elem, "force", attrib=fs_attrib)

    for ts in metadata.touch_sensors:
        site_elem = find_site(body_name=ts.body_name, site_name=ts.site_name, site_metadata=metadata.sites)

        # Add the touch sensor element
        ts_name = ts.name if ts.name else f"{ts.site_name}_touch"
        ts_attrib = {
            "name": ts_name,
            "site": ts.site_name,
        }
        if ts.noise is not None:
            ts_attrib["noise"] = str(ts.noise)

        ET.SubElement(sensor_elem, "touch", attrib=ts_attrib)

    # Save changes
    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_path", type=Path)
    args = parser.parse_args()

    add_sensors(args.mjcf_path, "base")


if __name__ == "__main__":
    # python -m urdf2mjcf.postprocess.add_sensors
    main()
