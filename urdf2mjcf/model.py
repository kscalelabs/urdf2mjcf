"""Defines the Pydantic model for the URDF to MJCF conversion."""

import enum
from typing import Literal

from pydantic import BaseModel

Angle = Literal["radian", "degree"]


class CollisionParams(BaseModel):
    condim: int = 3
    contype: int = 0
    conaffinity: int = 1
    solimp: list[float] = [0.99, 0.999, 0.00001]
    solref: list[float] = [0.005, 1.0]
    friction: list[float] = [1.0, 0.01, 0.01]


class JointMetadata(BaseModel):
    actuator_type: str | None = None
    id: int
    nn_id: int
    kp: float
    kd: float
    soft_torque_limit: float

    @classmethod
    def from_dict(cls, data: dict) -> "JointMetadata":
        """Create JointParam from a plain dictionary."""
        return cls(**data)

    class Config:
        extra = "forbid"


class ActuatorMetadata(BaseModel):
    actuator_type: str
    sysid: str = ""
    max_torque: float | None = None
    max_velocity: float | None = None
    armature: float | None = None
    damping: float | None = None
    frictionloss: float | None = None
    vin: float | None = None
    kt: float | None = None
    R: float | None = None
    max_pwm: float | None = None
    error_gain: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ActuatorMetadata":
        """Create ActuatorParam from a plain dictionary."""
        return cls.model_validate(data)


class ImuSensor(BaseModel):
    body_name: str
    pos: list[float] | None = None
    rpy: list[float] | None = None
    acc_noise: float | None = None
    gyro_noise: float | None = None
    mag_noise: float | None = None


class CameraSensor(BaseModel):
    name: str
    mode: str
    pos: list[float] | None = None
    rpy: list[float] | None = None
    fovy: float = 45.0


class ForceSensor(BaseModel):
    """Represents a force sensor attached to a site."""

    body_name: str
    site_name: str
    name: str | None = None
    noise: float | None = None


class ExplicitFloorContacts(BaseModel):
    """Add explicit floor contacts."""

    contact_links: list[str]
    class_name: str = "collision"


class CollisionType(enum.Enum):
    BOX = enum.auto()
    PARALLEL_CAPSULES = enum.auto()
    CORNER_SPHERES = enum.auto()
    SINGLE_SPHERE = enum.auto()


class CollisionGeometry(BaseModel):
    name: str
    collision_type: CollisionType
    sphere_radius: float = 0.01
    axis_order: tuple[int, int, int] = (0, 1, 2)
    flip_axis: bool = False
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0


class ConversionMetadata(BaseModel):
    freejoint: bool = True
    collision_params: CollisionParams = CollisionParams()
    joint_name_to_metadata: dict[str, JointMetadata] | None = None
    actuator_type_to_metadata: dict[str, ActuatorMetadata] | None = None
    imus: list[ImuSensor] = []
    cameras: list[CameraSensor] = [
        CameraSensor(
            name="front_camera",
            mode="track",
            pos=[0, 2.0, 0.5],
            rpy=[90.0, 0.0, 180.0],
            fovy=90,
        ),
        CameraSensor(
            name="side_camera",
            mode="track",
            pos=[-2.0, 0.0, 0.5],
            rpy=[90.0, 0.0, 270.0],
            fovy=90,
        ),
    ]
    force_sensors: list[ForceSensor] = []
    collision_geometries: list[CollisionGeometry] | None = None
    explicit_contacts: ExplicitFloorContacts | None = None
    remove_redundancies: bool = True
    floating_base: bool = True
    maxhullvert: int | None = None
    angle: Angle = "radian"
    floor_name: str = "floor"
    add_floor: bool = False
    backlash: float | None = None
    backlash_damping: float = 0.01
    height_offset: float = 0.0
    visualize_collision_meshes: bool = True

    class Config:
        extra = "forbid"
