"""Defines the Pydantic model for the URDF to MJCF conversion."""

from pydantic import BaseModel


class JointParam(BaseModel):
    kp: float | None = None
    kd: float | None = None

    class Config:
        extra = "forbid"


class JointParamsMetadata(BaseModel):
    suffix_to_pd_params: dict[str, JointParam] = {}
    default: JointParam | None = None

    class Config:
        extra = "forbid"


class ImuSensor(BaseModel):
    """Configuration for an IMU sensor.

    Attributes:
        site_name: Name of the site to attach the IMU to
        pos: Position relative to site frame, in the form [x, y, z]
        quat: Quaternion relative to site frame, in the form [w, x, y, z]
    """

    site_name: str
    pos: list[float] = [0.0, 0.0, 0.0]
    quat: list[float] = [1.0, 0.0, 0.0, 0.0]
    acc_noise: float | None = None
    gyro_noise: float | None = None
    mag_noise: float | None = None


class ConversionMetadata(BaseModel):
    """Configuration for URDF to MJCF conversion.

    Attributes:
        joint_params: Optional PD gains metadata for joints
        imus: Optional list of IMU sensor configurations
        merge_fixed: If True, merge fixed joints
        floating_base: If True, add a floating base to the MJCF model
        remove_base_inertial: If True, remove the base inertial element from
            the MJCF model
    """

    joint_params: JointParamsMetadata | None = None
    imus: list[ImuSensor] = []
    merge_fixed: bool = False
    floating_base: bool = True
    remove_base_inertial: bool = False

    class Config:
        extra = "forbid"
