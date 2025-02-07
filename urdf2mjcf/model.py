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
        link_name: Name of the link to attach the IMU to
        pos: Position relative to link frame, in the form [x, y, z]
        quat: Quaternion relative to link frame, in the form [w, x, y, z]
    """

    link_name: str
    pos: list[float] = [0.0, 0.0, 0.0]
    quat: list[float] = [1.0, 0.0, 0.0, 0.0]
    acc_noise: float | None = None
    gyro_noise: float | None = None
    mag_noise: float | None = None


class ConversionMetadata(BaseModel):
    """Configuration for URDF to MJCF conversion.

    Attributes:
        joint_params: Optional PD gains metadata for joints
        cameras: Optional list of camera sensor configurations
        imus: Optional list of IMU sensor configurations
    """

    joint_params: JointParamsMetadata | None = None
    imus: list[ImuSensor] = []

    class Config:
        extra = "forbid"
