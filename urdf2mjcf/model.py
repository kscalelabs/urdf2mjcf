"""Defines the Pydantic model for the URDF to MJCF conversion."""

from pydantic import BaseModel


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


class FeetSpheresParams(BaseModel):
    foot_links: list[str]
    sphere_radius: float


class ConversionMetadata(BaseModel):
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
    feet_spheres: FeetSpheresParams | None = None
    remove_redundancies: bool = True
    floating_base: bool = True

    class Config:
        extra = "forbid"
