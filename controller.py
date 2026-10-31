from __future__ import annotations

# The devs of cvxpy are absolute legends for making this
import cvxpy as cp
import numpy as np

from pathlib import Path

from dataclasses import dataclass
from typing import Optional, Callable, ParamSpec, TypeVar
from functools import wraps
from enum import Enum, auto
from collections import deque

import logging
import math

from inference import DefenseInferenceStack, DefenseModelOutput
from dataloader import Frame
import config

@dataclass
class Vec2:
    x: float
    y: float

    @staticmethod
    def right():
        return Vec2(0, 1)
    
    @staticmethod
    def forward():
        return Vec2(1, 0)

    @staticmethod
    def zero():
        return Vec2(0, 0)

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: Vec2):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int|float):
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other: int|float):
        return Vec2(self.x / other, self.y / other)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def astuple(self):
        return self.x, self.y

    def dot(self, other: Vec2):
        return math.sumprod(self, other)

class ControllerState(Enum):
    SEARCHING = auto()
    BLOCKING = auto()
    INTERCEPTING = auto()

@dataclass
class ControllerAction:
    state: ControllerState
    accel: Optional[tuple[float, float]] # Field-relative (x, y) OR None to represent "stay stationary"
    rot_accel: float # Radians

    @staticmethod
    def idle() -> ControllerAction:
        return ControllerAction(ControllerState.SEARCHING, None, 0)

class ControllerObjective(Enum):
    STOP_POSITIVE_X = auto()
    STOP_NEGATIVE_X = auto()

@dataclass
# Units are m and rad
class RobotConstraints:
    MAX_SPEED = 5.25
    MAX_ACCEL = 5.25
    MAX_ANGULAR_ACCEL = 20.0
    MAX_ANGULAR_SPEED = 5.5

# My first decorator!
P = ParamSpec("P")
R = TypeVar("R")
def catcherrors(f: Callable[P, R]) -> Callable[P, Optional[R]]:
    """ In the event of errors: catches, logs, and returns None """
    @wraps(f)
    def fn(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
        try:
            return f(*args, **kwargs)
        except Exception:
            logging.exception("Caught error")
            return None
    return fn

class Solver:
    def __init__(self, horizon: int, robot_constraints: RobotConstraints):
        dt = config.TIMESTEP / 1000 # Convert to seconds

        ############ VARIABLES ############
        self.pos = cp.Variable((2, horizon + 1)) # m
        self.vel = cp.Variable((2, horizon + 1)) # m/s
        self.acc = cp.Variable((2, horizon))     # m/s/s

        self.angle = cp.Variable(horizon + 1) # Rad
        self.omega = cp.Variable(horizon + 1) # Rad/s
        self.alpha = cp.Variable(horizon)     # Rad/s/s

        # Initial state (updated every tick)
        self.p0 = cp.Parameter(2)
        self.v0 = cp.Parameter(2)
        self.theta0 = cp.Parameter()
        self.omega0 = cp.Parameter()

        # Targets (updated every tick)
        self.goal_pos = cp.Parameter(2)
        self.goal_vel = cp.Parameter(2)
        self.goal_angle = cp.Parameter()
        self.previous_acc = cp.Parameter(2, value=np.zeros(2))

        ############ CONSTRAINTS AND COST ############

        # Squaring the scale 8 gives a lateral weight of 64
        axis_scale = np.array([1.0, 8.0])
        self.cost = 80.0 * cp.sum_squares(cp.multiply(axis_scale, self.pos[:, -1] - self.goal_pos))
        self.cost += 2.0 * cp.sum_squares(cp.multiply(axis_scale, self.vel[:, -1] - self.goal_vel))
        self.cost += 0.00025 * cp.sum_squares(self.acc)
        self.cost += 0.00015 * cp.sum_squares(self.acc[:, 0] - self.previous_acc)
        if horizon > 1:
            self.cost += 0.015 * cp.sum_squares(self.acc[:, 1:] - self.acc[:, :-1])

        # Preserve your separate angular objective and effort penalty
        self.cost += 65.0 * cp.abs(self.angle[-1] - self.goal_angle) # We care a LOT about angle (especially when blocking so they can't pivot around us)
        self.cost += 5.0 * cp.abs(self.omega[-1]) # Just always try to have 0 rotational velocity at the end of a frame
        self.cost += (0.008 / (1.0 / (2 * math.pi) ** 2)) * cp.sum_squares(self.alpha / (2 * math.pi))

        # The first frame must contain the current robot state
        self.constraints = [
            self.pos[:, 0] == self.p0,
            self.vel[:, 0] == self.v0,
            self.angle[0] == self.theta0,
            self.omega[0] == self.omega0,
        ]

        for i in range(horizon):
            self.constraints += [
                # Simple integrals here
                self.pos[:, i+1] == self.pos[:, i] + self.vel[:, i]*dt + (1/2)*self.acc[:, i]*(dt**2),
                self.vel[:, i+1] == self.vel[:, i] + self.acc[:, i]*dt,

                self.angle[i+1] == self.angle[i] + self.omega[i]*dt + (1/2)*self.alpha[i]*(dt**2),
                self.omega[i+1] == self.omega[i] + self.alpha[i]*dt,


                cp.norm(self.vel[:, i+1]) <= robot_constraints.MAX_SPEED,
                cp.norm(self.acc[:, i]) <= robot_constraints.MAX_ACCEL,

                cp.abs(self.omega[i+1]) <= robot_constraints.MAX_ANGULAR_SPEED,
                cp.abs(self.alpha[i]) <= robot_constraints.MAX_ANGULAR_ACCEL
            ]


        ############ PROBLEM ############

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints) # pyright: ignore[reportArgumentType]
    
    def __call__(self,
                 pos: Vec2,
                 vel: Vec2,
                 theta: float,
                 omega: float,
                 goal_pos: Vec2,
                 goal_vel: Vec2,
                 goal_angle: float
                ) -> Optional[tuple[Vec2, float]]:
        # Update parameters
        self.p0.value = [*pos]
        self.v0.value = [*vel]

        self.theta0.value = theta
        self.omega0.value = omega

        self.goal_pos.value = [*goal_pos]
        self.goal_vel.value = [*goal_vel]
        self.goal_angle.value = theta + math.remainder(goal_angle - theta, 2 * math.pi)

        # Solve the SAME problem to reuse old estimates
        self.problem.solve(solver=cp.CLARABEL, warm_start=True)

        # I'm in typehint hell
        if self.problem.status == cp.OPTIMAL or self.problem.status == cp.OPTIMAL_INACCURATE:
            acc = self.acc[:, 0]
            assert acc is not None and acc.value is not None
            ax, ay = acc.value
            assert ax is not None and ay is not None
            assert self.alpha[0].value is not None
            self.previous_acc.value = np.array([ax, ay], dtype=float)
            return Vec2(ax, ay), self.alpha[0].value.item()

        return None

class Controller:
    def __init__(self, model_path: Path, constraints: RobotConstraints=RobotConstraints()):
        self.constraints = constraints
        self.defense_inference_stack = DefenseInferenceStack(model_path)
        self.objective = ControllerObjective.STOP_POSITIVE_X
        horizon = round(config.SEARCH_LOOKAHEAD * 1000 / config.TIMESTEP)
        logging.info(f"Using {horizon} frame solver search length")
        self.solver = Solver(horizon, constraints)
        self.initial_pos = None
        self.opp_pos_history: deque[None|Vec2] = deque([None] * 10, maxlen=10)

        self.last_act = ControllerAction.idle()
        self.repeated_ticks = 0

    def reset(self, self_x: float, self_y: float):
        self.defense_inference_stack.reset()
        self.solver.previous_acc.value = np.zeros(2)
        self.initial_pos = Vec2(self_x, self_y)
        for _ in range(10):
            self.opp_pos_history.append(None)

        self.last_act = ControllerAction.idle()
        self.repeated_ticks = 0

    @property
    def frames(self):
        return self.defense_inference_stack.frames

    @property
    def last_frame(self):
        return self.frames[-1]

    def set_objective(self, new_objective: ControllerObjective):
        self.objective = new_objective

    @catcherrors
    def update(self, frame: Frame) -> ControllerAction:
        new_action = self._update_single_frame(frame)

        # If we were just doing something and randomly drop a frame which makes us search, we'll just execute the action again. Otherwise, it'll update next tick
        if new_action.state == ControllerState.SEARCHING and self.last_act.state != ControllerState.SEARCHING and self.repeated_ticks < config.MAX_DROPOUT_TICKS:
            self.repeated_ticks += 1
            return self.last_act

        self.last_act = new_action
        self.repeated_ticks = 0

        return new_action

    def _update_single_frame(self, frame: Frame) -> ControllerAction:
        """ Applies a single frame of updates, which doesn't handle random dropouts """
        self.defense_inference_stack.add_frame(frame)
        preds = self.defense_inference_stack.predict()

        # Got to take the sin/cos and convert back to radians
        self_angle = math.atan2(self.last_frame.rot_sin, self.last_frame.rot_cos)
        self_omega = frame.vel_angular

        assert self.initial_pos is not None, "Didn't call reset() before update() in Controller"

        ############  None (search)  ############
        self_pos = Vec2(self.last_frame.pos_x, self.last_frame.pos_z)
        self_vel = self._to_field_coords(self.last_frame.vel_fwd, self.last_frame.vel_right, self_angle)

        def build_search_action() -> ControllerAction:
            assert self.initial_pos is not None # It can't be by here, but pyright doesn't know that
            solution = self.solver(
                self_pos, self_vel, self_angle, self_omega,
                self.initial_pos, Vec2.zero(), self._get_search_rot()
            )
            if solution is not None:
                accel, rot_accel = solution
                return ControllerAction(ControllerState.SEARCHING, accel.astuple(), rot_accel)
            return ControllerAction(ControllerState.SEARCHING, None, self._get_search_rot() - self_angle) # Will naturally oscillate pretty bad since we're going from error in radians to acceleration but it should be rare

        if not any(p > config.CONF_THRESHOLD for p in preds.ray_hit_confidences):
            self.opp_pos_history.append(None) # No opponent means we cannot predict position
            return build_search_action()

        ############ Intercept/block ############
        # Solve target position by calculating center in field/robot-relative polar (hybrid) and converting back to euclidean
        opp_theta, opp_r = self._get_opponent_position_polar(self_angle, preds)
        opp_pos = Vec2(opp_r * math.cos(opp_theta), opp_r * math.sin(opp_theta)) + self_pos

        self.opp_pos_history.append(opp_pos)

        opp_vel = self._estimate_opp_vel()

        self_to_opp = opp_pos - self_pos
        angle_toward_opp = math.atan2(self_to_opp.y, self_to_opp.x)

        was_just_blocking_mult = 2 if self.last_act.state == ControllerState.BLOCKING else 1

        # First check if we should block
        # Note that we give larger X/Y error bounds if we were just blocking (to handle noise better)
        if abs(self_pos.y - opp_pos.y) < config.BLOCKING.MAX_Y_ERROR * was_just_blocking_mult  and \
            abs(self_pos.x - opp_pos.x) < config.BLOCKING.MAX_X_DISTANCE and \
            self._get_blocked_opp_vel().dot(self_pos - opp_pos) > 0:
                # If we should block, we'll assume swerve is rotated inline with the Y axis at rest so we shouldn't send X commands.
                # Given this, we'll pass the same target X as our current X
                solution = self.solver(
                    self_pos, self_vel, self_angle, self_omega,
                    Vec2(self_pos.x, opp_pos.y), Vec2.zero(), self._get_intercept_rot()
                )
                # If the solution exists, return
                if solution is not None:
                    accel, rot_accel = solution
                    return ControllerAction(ControllerState.BLOCKING, (0, accel.y), rot_accel) # It's technically possible that accel.x is nonzero if the opponent is pushing us, but we'll 0 it anyway because spinning the wheels risks field damage
                else:
                    return ControllerAction(ControllerState.BLOCKING, (0, 1e-10), 0) # Just a tiny hint of y movement to try to get swerve to align if the outer controller doesn't do so automatically

        # Otherwise, solve for intercept
        expected_time_of_intercept = config.SEARCH_LOOKAHEAD / 2
        target_pos = opp_pos + opp_vel * expected_time_of_intercept

        # Note that here we only try to match Y values (get in the way)
        solution = self.solver(
            self_pos, self_vel, self_angle, self_omega,
            Vec2(self.initial_pos.x, target_pos.y), Vec2.zero(), (angle_toward_opp + self._get_intercept_rot()) / 2
        )

        # If the solution exists, return
        if solution is not None:
            accel, rot_accel = solution
            return ControllerAction(ControllerState.INTERCEPTING, accel.astuple(), rot_accel)

        # If we get here, the solver failed to find a valid solution, so we just fall back to searching in hopes of finding a new target
        return build_search_action()

    def _estimate_opp_vel(self) -> Vec2:
        if self.opp_pos_history[-1] is None or self.opp_pos_history[-2] is None:
            return Vec2.zero()

        curr_pos = self.opp_pos_history[-1]
        last_pos = self.opp_pos_history[-2]

        return (curr_pos - last_pos) / (config.TIMESTEP / 1000)

    def _get_blocked_opp_vel(self) -> Vec2:
        """ Returns a vector in the direction which the opponent is expected to travel """
        return Vec2.forward() * (-1 if self.objective == ControllerObjective.STOP_NEGATIVE_X else 1)

    def _get_intercept_rot(self) -> float:
        """ Return the target intercept rotation for the given objective in rads """
        return 0 if self.objective == ControllerObjective.STOP_NEGATIVE_X else math.pi

    def _get_search_rot(self) -> float:
        """ Return the idle/"looking for target" rotation for the given objective in rads """
        return 0 if self.objective == ControllerObjective.STOP_NEGATIVE_X else math.pi
    
    def _fov_for(self, ray: int) -> float:
        """ Return the radians from forward for the given ray by index """
        # Rays are indexed from right to left so values should go from -fov to +fov
        normalized = 1 - 2 * ray / (config.EXPECTED_RAYS - 1)
        return math.radians(normalized * config.FOV_PER_SIDE)

    def _get_opponent_position_polar(self, self_angle: float, preds: DefenseModelOutput) -> tuple[float, float]:
        """ Return the opponent's rotation (with field relative rotation) where positive means right, and distance to get there from the current robot """
        num_consecuitive_hits = [0] * len(preds.ray_hit_confidences)
        for i, confidence in enumerate(preds.ray_hit_confidences):
            if confidence > config.CONF_THRESHOLD:
                num_consecuitive_hits[i] = num_consecuitive_hits[i - 1] + 1

        largest_hit_size = max(num_consecuitive_hits)
        center_of_largest_hit = num_consecuitive_hits.index(largest_hit_size) - largest_hit_size // 2
        c1, c2 = center_of_largest_hit, center_of_largest_hit + (1 if largest_hit_size % 2 == 0 else 0) # c1 and c2 are the 2 closest raycasts to the center so we avg them
        
        angle = self_angle - (self._fov_for(c1) + self._fov_for(c2)) / 2
        distance = (self.last_frame.raycast_distances[c1] + self.last_frame.raycast_distances[c2]) / 2 + config.EST_BUPER_TO_CENTER

        return angle, distance

    @staticmethod
    def _to_field_coords(forward: float, right: float, yaw: float):
        return Vec2(
            math.cos(yaw) * forward + math.sin(yaw) * right,
            math.sin(yaw) * forward - math.cos(yaw) * right,
        )
