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
    pos: list[float] = [0.0, 0.0, 0.0]
    quat: list[float] = [1.0, 0.0, 0.0, 0.0]
    acc_noise: float | None = None
    gyro_noise: float | None = None
    mag_noise: float | None = None


class CameraSensor(BaseModel):
    name: str
    mode: str
    pos: list[float] = [0.0, 0.0, 0.0]
    quat: list[float] = [1.0, 0.0, 0.0, 0.0]
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
            name="tracking_camera",
            mode="track",
            pos=[0, -2.0, 1.0],
            quat=[0.7071, 0.3827, 0, 0],
            fovy=90,
        ),
    ]
    feet_spheres: FeetSpheresParams | None = None
    remove_redundancies: bool = True
    floating_base: bool = True

    class Config:
        extra = "forbid"
