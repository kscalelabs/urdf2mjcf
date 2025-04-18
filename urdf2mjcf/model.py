"""Defines the Pydantic model for the URDF to MJCF conversion."""

from typing import Literal

from pydantic import BaseModel

Angle = Literal["radian", "degree"]


class CollisionParams(BaseModel):
    friction: list[float] = [0.8, 0.02, 0.01]
    condim: int = 6


class JointParam(BaseModel):
    name: str
    suffixes: list[str]
    armature: float | None = None
    frictionloss: float | None = None
    actuatorfrc: float | None = None

    class Config:
        extra = "forbid"


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


class ConversionMetadata(BaseModel):
    freejoint: bool = True
    collision_params: CollisionParams = CollisionParams()
    joint_params: list[JointParam] | None = None
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
    flat_feet_links: list[str] | None = None
    explicit_floor_contacts: list[str] | None = None
    remove_redundancies: bool = True
    floating_base: bool = True
    maxhullvert: int | None = None
    angle: Angle = "radian"

    class Config:
        extra = "forbid"
