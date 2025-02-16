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


class CameraSensor(BaseModel):
    """Configuration for a camera sensor.

    Attributes:
        site_name: Name of the site to attach the camera to
        pos: Position relative to site frame, in the form [x, y, z]
        quat: Quaternion relative to site frame, in the form [w, x, y, z]
        fovy: Field of view in degrees
    """

    name: str
    mode: str
    pos: list[float] = [0.0, 0.0, 0.0]
    quat: list[float] = [1.0, 0.0, 0.0, 0.0]
    fovy: float = 45.0


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
    cameras: list[CameraSensor] = [
        CameraSensor(
            name="tracking_camera",
            mode="track",
            pos=[0, -2.0, 1.0],
            quat=[0.7071, 0.3827, 0, 0],
            fovy=90,
        ),
    ]
    remove_fixed_joints: bool = False
    remove_redundancies: bool = True
    floating_base: bool = True

    class Config:
        extra = "forbid"
