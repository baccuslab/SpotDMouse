from dataclasses import dataclass
from typing import Tuple

import jax
from jax import numpy as jp


def domain_randomize(
    sys,
    rng,
    friction_range: Tuple = (0.6, 1.4),
    kp_multiplier_range: Tuple = (0.75, 1.25),
    kd_multiplier_range: Tuple = (0.5, 2.0),
    body_com_x_shift_range: Tuple = (-0.03, 0.03),
    body_com_y_shift_range: Tuple = (-0.01, 0.01),
    body_com_z_shift_range: Tuple = (-0.02, 0.02),
    body_inertia_scale_range: Tuple = (0.7, 1.3),
    body_mass_scale_range: Tuple = (0.7, 1.3),
):
    """Randomizes the friction, actuator kp, & actuator kd

    TODO: What is the dimension of rng? The number of environments?
    """

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        rng, key_kp, key_kd = jax.random.split(rng, 3)
        kp = (
            jax.random.uniform(
                key_kp,
                (1,),
                minval=kp_multiplier_range[0],
                maxval=kp_multiplier_range[1],
            )
            * sys.actuator_gainprm[:, 0]
        )
        kd = jax.random.uniform(
            key_kd,
            (1,),
            minval=kd_multiplier_range[0],
            maxval=kd_multiplier_range[1],
        ) * (-sys.actuator_biasprm[:, 2])

        gain = sys.actuator_gainprm.at[:, 0].set(kp)
        bias = sys.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)

        rng, key_com = jax.random.split(rng)
        body_com_shift = jax.random.uniform(
            key_com,
            (3,),
            minval=jp.array([
                body_com_x_shift_range[0],
                body_com_y_shift_range[0],
                body_com_z_shift_range[0],
            ]),
            maxval=jp.array([
                body_com_x_shift_range[1],
                body_com_y_shift_range[1],
                body_com_z_shift_range[1],
            ]),
        )
        body_com = sys.body_ipos.at[1].set(sys.body_ipos[1] + body_com_shift)

        # TODO(nathankau) think if we want to scale inertia uniformly or not
        # TODO(nathankau) do we want to randomize all links inertia or just main body?
        rng, key_inertia = jax.random.split(rng)
        body_inertia_scale = jax.random.uniform(
            key_inertia,
            sys.body_inertia.shape,
            minval=body_inertia_scale_range[0],
            maxval=body_inertia_scale_range[1],
        )
        body_inertia = sys.body_inertia * body_inertia_scale

        rng, key_mass = jax.random.split(rng)
        body_mass_scale = jax.random.uniform(
            key_mass,
            sys.body_mass.shape,
            minval=body_mass_scale_range[0],
            maxval=body_mass_scale_range[1],
        )
        body_mass = sys.body_mass * body_mass_scale

        return friction, gain, bias, body_com, body_inertia, body_mass

    friction, gain, bias, body_com, body_inertia, body_mass = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "body_ipos": 0,
        "body_inertia": 0,
        "body_mass": 0,
    })

    sys = sys.tree_replace({
        "geom_friction": friction,
        "actuator_gainprm": gain,
        "actuator_biasprm": bias,
        "body_ipos": body_com,
        "body_inertia": body_inertia,
        "body_mass": body_mass,
    })

    return sys, in_axes


@dataclass
class StartPositionRandomization:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def small_quaternion(rng, max_angle_deg=30, max_yaw_deg=180):
    """
    Returns a quaternion with random pitch and roll in [-max_angle_deg, max_angle_deg]
    (in degrees) and yaw in [-max_yaw_deg, max_yaw_deg], then converted to radians and into quaternion form.

    Args:
        rng: JAX random key.
        max_angle_deg (float): Maximum magnitude of pitch/roll in degrees.
        max_yaw_deg (float): Maximum magnitude of yaw in degrees.

    Returns:
        jax.numpy array: Shape (4,). The quaternion (w, x, y, z).
    """

    rng, key_pitch, key_roll, key_yaw = jax.random.split(rng, 4)

    # Random pitch, roll, and yaw in [-max_angle_deg, max_angle_deg]
    pitch_deg = (jax.random.uniform(key_pitch, ()) * 2 - 1) * max_angle_deg
    roll_deg = (jax.random.uniform(key_roll, ()) * 2 - 1) * max_angle_deg
    yaw_deg = (jax.random.uniform(key_yaw, ()) * 2 - 1) * max_yaw_deg

    # Convert degrees to radians
    pitch_rad = pitch_deg * jp.pi / 180.0
    roll_rad = roll_deg * jp.pi / 180.0
    yaw_rad = yaw_deg * jp.pi / 180.0

    # Half angles
    half_pitch = pitch_rad / 2
    half_roll = roll_rad / 2
    half_yaw = yaw_rad / 2

    # cosines and sines of half angles
    cr = jp.cos(half_roll)
    sr = jp.sin(half_roll)
    cp = jp.cos(half_pitch)
    sp = jp.sin(half_pitch)
    cy = jp.cos(half_yaw)
    sy = jp.sin(half_yaw)

    # Convert Euler angles -> Quaternion
    # Roll (X), Pitch (Y), Yaw (Z) in intrinsic rotation order
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Stack to form quaternion
    q = jp.array([w, x, y, z])

    # Normalize quaternion (though it should already be nearly normalized)
    q = q / jp.linalg.norm(q)

    return q


def random_z_rotation_quaternion(rng):
    """Generates a random quaternion with a random yaw angle."""
    yaw = jax.random.uniform(rng, (1,), minval=-jp.pi, maxval=jp.pi)
    cos_yaw = jp.cos(yaw / 2)
    sin_yaw = jp.sin(yaw / 2)
    return jp.concatenate((cos_yaw, jp.zeros(2), sin_yaw))


def randomize_qpos(qpos: jp.array, start_position_config: StartPositionRandomization, rng):
    """Return qpos with randomized position of first body. Do not use rng again!"""

    rng, key_pos, key_yaw = jax.random.split(rng, 3)
    qpos = qpos.at[:3].set(
        jax.random.uniform(
            key_pos,
            shape=(3,),
            minval=jp.array((
                start_position_config.x_min,
                start_position_config.y_min,
                start_position_config.z_min,
            )),
            maxval=jp.array((
                start_position_config.x_max,
                start_position_config.y_max,
                start_position_config.z_max,
            )),
        )
    )
    random_yaw_quat = random_z_rotation_quaternion(key_yaw)
    qpos = qpos.at[3:7].set(random_yaw_quat)
    return qpos
