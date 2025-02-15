"""Defines the Pydantic model for the URDF to MJCF conversion."""

from pydantic import BaseModel


class JointParam(BaseModel):
    name: str
    suffixes: list[str]
    armature: float | None = None
    frictionloss: float | None = None
    actuatorfrc: float | None = None
    kp: float | None = None
    kd: float | None = None

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
        remove_fixed_joints: If True, convert fixed child bodies into sites on
            their parent bodies
        floating_base: If True, add a floating base to the MJCF model
    """

    joint_params: list[JointParam] | None = None
    imus: list[ImuSensor] = []
    remove_fixed_joints: bool = False
    remove_redundancies: bool = True
    floating_base: bool = True

    class Config:
        extra = "forbid"
