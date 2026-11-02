"""ROS-agnostic, observation-only FAST-LIVO2 -> MissionSnapshot v2 bridge.

Important: 2D ray projection is a *debug diagnostic* only. It cannot infer
safe 3D UAV flight corridors, reliable frontiers, semantics, or usable energy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, floor, hypot, isfinite, sqrt
from typing import Sequence

from interfaces import MissionSpec, validate_spec_snapshot
from interfaces.state import GoalProgress, GoalStatus, MissionSnapshot, Position, Resources
from visualization.adapters import frame_from_observed
from visualization.frame import VisualPoint, VisualPlanner, VisualizationFrame


def _finite(*values: float):
    if not all(isfinite(v) for v in values):
        raise ValueError('Pose/point contains NaN or infinity')


def _unit_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    _finite(*q)
    n = sqrt(sum(v*v for v in q))
    if n < 1e-9:
        raise ValueError('Invalid zero quaternion')
    if not .8 <= n <= 1.2:
        raise ValueError('Quaternion norm is implausible')
    return tuple(v / n for v in q)


def _quat_rotate(q: Sequence[float], p: Sequence[float]) -> tuple[float, float, float]:
    """Rotate xyz using quaternion in ROS xyzw convention."""
    x, y, z, w = q
    px, py, pz = p
    uvx, uvy, uvz = y*pz-z*py, z*px-x*pz, x*py-y*px
    uuvx, uuvy, uuvz = y*uvz-z*uvy, z*uvx-x*uvz, x*uvy-y*uvx
    return (px + 2*(w*uvx+uuvx), py + 2*(w*uvy+uuvy), pz + 2*(w*uvz+uuvz))


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


@dataclass(frozen=True)
class PoseSample:
    stamp_s: float
    frame_id: str
    child_frame_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        _finite(self.stamp_s, *self.position)
        if self.stamp_s <= 0 or not self.frame_id or not self.child_frame_id:
            raise ValueError('Pose must have nonzero timestamp and explicit frames')
        object.__setattr__(self, 'quaternion_xyzw', _unit_quat(self.quaternion_xyzw))


@dataclass(frozen=True)
class CloudSample:
    stamp_s: float
    frame_id: str
    xyz: tuple[tuple[float, float, float], ...]
    def __post_init__(self):
        _finite(self.stamp_s)
        if self.stamp_s <= 0 or not self.frame_id:
            raise ValueError('Cloud must have nonzero stamp and frame')
        if len(self.xyz) > 100000:
            raise ValueError('Input cloud sample must be subsampled before bridging')
        for p in self.xyz:
            if len(p) != 3:
                raise ValueError('Point must have xyz')
            _finite(*p)


@dataclass(frozen=True)
class CameraSample:
    """Image metadata only. Images are not transmitted or classified in Step 5."""
    stamp_s: float
    frame_id: str
    width: int
    height: int
    encoding: str

    def __post_init__(self):
        _finite(self.stamp_s)
        if self.stamp_s <= 0 or not self.frame_id or self.width <= 0 or self.height <= 0:
            raise ValueError('Invalid camera timestamp/frame/dimensions')


@dataclass(frozen=True)
class SensorToBody:
    """Calibrated T_body_sensor: x_body=R_body_sensor*x_sensor+t_body_sensor."""
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        _finite(*self.translation)
        object.__setattr__(self, 'rotation_xyzw', _unit_quat(self.rotation_xyzw))


@dataclass(frozen=True)
class BridgeConfig:
    width: int = 100
    height: int = 100
    resolution_m: float = 0.5
    origin_x_m: float = -25.0
    origin_y_m: float = -25.0
    max_range_m: float = 12.0
    min_range_m: float = 0.5
    min_relative_z_m: float = -0.6
    max_relative_z_m: float = 1.2
    max_interpolation_gap_s: float = 0.5
    max_odom_age_s: float = 1.0
    max_pending_clouds: int = 40
    max_points_per_cloud: int = 512
    pose_history_size: int = 2000
    sensor_to_body: SensorToBody | None = None
    sensor_frame_id: str | None = None

    def __post_init__(self):
        if not (2 <= self.width <= 800 and 2 <= self.height <= 800):
            raise ValueError('Grid dimensions must be 2..800')
        if not (0 < self.resolution_m <= 10 and 0 <= self.min_range_m < self.max_range_m
                and self.max_interpolation_gap_s > 0 and self.max_odom_age_s > 0):
            raise ValueError('Invalid grid, range, or timestamp settings')
        if self.min_relative_z_m >= self.max_relative_z_m:
            raise ValueError('Invalid relative z window')
        if min(self.max_pending_clouds, self.max_points_per_cloud, self.pose_history_size) < 2:
            raise ValueError('Buffer limits must be >= 2')
        _finite(self.origin_x_m, self.origin_y_m)
        if self.sensor_to_body and not self.sensor_frame_id:
            raise ValueError('Provide --sensor-frame with an extrinsic to prevent applying it to the wrong LiDAR frame')


class ObservedGrid:
    """Minimal bounded 2D ray projection, **not a navigation occupancy map**."""
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.cells = [[-1 for _ in range(cfg.width)] for _ in range(cfg.height)]
        self.update_count = 0

    def index(self, x: float, y: float) -> tuple[int, int] | None:
        col = floor((x-self.cfg.origin_x_m)/self.cfg.resolution_m)
        row = floor((y-self.cfg.origin_y_m)/self.cfg.resolution_m)
        if 0 <= row < self.cfg.height and 0 <= col < self.cfg.width:
            return (row, col)
        return None

    def ray(self, src: tuple[float, float, float], hit: tuple[float, float, float], occupied: bool):
        """Rasterize the XY ray; no free-space claims outside configured grid."""
        length = hypot(hit[0]-src[0], hit[1]-src[1])
        steps = max(1, ceil(length/self.cfg.resolution_m*2))
        visited = []
        for i in range(steps+1):
            t = i/steps
            idx = self.index(src[0]+(hit[0]-src[0])*t, src[1]+(hit[1]-src[1])*t)
            if idx is not None and (not visited or idx != visited[-1]):
                visited.append(idx)
        if not visited:
            return
        for r,c in visited[:-1] if occupied else visited:
            if self.cells[r][c] != 1:
                self.cells[r][c] = 0
        if occupied:
            r,c = visited[-1]
            self.cells[r][c] = 1
        self.update_count += 1

    def as_tuple(self):
        return tuple(tuple(row) for row in self.cells)

    def known_cells(self):
        return sum(cell != -1 for row in self.cells for cell in row)


class RosObservationBridge:
    def __init__(self, spec: MissionSpec, cfg: BridgeConfig | None = None):
        self.spec = spec
        self.cfg = cfg or BridgeConfig()
        self.grid = ObservedGrid(self.cfg)
        self.poses: deque[PoseSample] = deque(maxlen=self.cfg.pose_history_size)
        self.pending: list[CloudSample] = []
        self.frame_id: str | None = None
        self.child_frame_id: str | None = None
        self.home: tuple[float, float, float] | None = None
        self.path: list[VisualPoint] = []
        self.state_seq = 0
        self.map_version = 0
        self.accepted_clouds = 0
        self.dropped_clouds = 0
        self.rejection_reasons: dict[str, int] = {}
        self.last_processed_stamp_s: float | None = None
        self.initial_sensor_frame: str | None = None
        self.camera_frames = 0
        self.last_camera_stamp_s: float | None = None
        self.camera_frame_id: str | None = None

    def _reject(self, reason):
        self.dropped_clouds += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0)+1

    def receive_pose(self, pose: PoseSample):
        frame = pose.frame_id.lstrip('/')
        child = pose.child_frame_id.lstrip('/')
        if (self.frame_id and frame != self.frame_id) or (self.child_frame_id and child != self.child_frame_id):
            raise ValueError(f'Odometry frame changed: {frame}/{child}')
        if self.poses and pose.stamp_s <= self.poses[-1].stamp_s:
            raise ValueError('Out-of-order or duplicate odometry stamp')
        self.frame_id, self.child_frame_id = frame, child
        self.poses.append(pose)
        if self.home is None:
            self.home = pose.position
        self.path.append(VisualPoint(x=pose.position[0], y=pose.position[1]))
        if len(self.path) > 2000: self.path.pop(0)  # bounded dashboard history
        self.state_seq += 1
        self._flush_clouds()

    def receive_camera(self, image: CameraSample):
        # Metadata-only. No trusted semantic evidence or image storage here.
        if self.camera_frame_id is None:
            self.camera_frame_id = image.frame_id
        if self.camera_frame_id != image.frame_id:
            raise ValueError('Camera frame changed')
        if self.last_camera_stamp_s is not None and image.stamp_s <= self.last_camera_stamp_s:
            raise ValueError('Camera timestamp regressed')
        self.last_camera_stamp_s = image.stamp_s
        self.camera_frames += 1
        self.state_seq += 1

    def _pose_at(self, stamp_s: float) -> PoseSample | None:
        if not self.poses or stamp_s < self.poses[0].stamp_s or stamp_s > self.poses[-1].stamp_s:
            return None
        # Search backwards; last pose is usually closest to the scan.
        seq = self.poses
        for i in range(len(seq)-1, -1, -1):
            a = seq[i]
            if a.stamp_s == stamp_s:
                return a
            if a.stamp_s < stamp_s:
                b = seq[i+1]
                if b.stamp_s-a.stamp_s > self.cfg.max_interpolation_gap_s:
                    return None
                u = (stamp_s-a.stamp_s)/(b.stamp_s-a.stamp_s)
                xyz = tuple((1-u)*x+u*y for x,y in zip(a.position,b.position))
                bq = b.quaternion_xyzw
                if sum(x*y for x,y in zip(a.quaternion_xyzw,bq)) < 0:
                    bq = tuple(-v for v in bq)
                q = _unit_quat(tuple((1-u)*x+u*y for x,y in zip(a.quaternion_xyzw,bq)))
                return PoseSample(stamp_s, a.frame_id, a.child_frame_id, xyz, q)
        return None

    def receive_cloud(self, cloud: CloudSample):
        if len(self.pending) >= self.cfg.max_pending_clouds:
            self.pending.pop(0)
            self._reject('PENDING_OVERFLOW')
        self.pending.append(cloud)
        self.pending.sort(key=lambda x: x.stamp_s)
        self._flush_clouds()

    def _flush_clouds(self):
        if not self.poses:
            return
        hold = []
        for cloud in self.pending:
            if cloud.stamp_s > self.poses[-1].stamp_s:
                hold.append(cloud)
            elif cloud.stamp_s < self.poses[0].stamp_s:
                self._reject('NO_POSE_BRACKET')
            else:
                pose = self._pose_at(cloud.stamp_s)
                if pose is None:
                    self._reject('POSE_GAP_TOO_LARGE')
                else:
                    self._integrate(cloud, pose)
        self.pending = hold

    def flush_end_of_stream(self):
        for _ in self.pending:
            self._reject('UNPAIRED_AT_END')
        self.pending.clear()

    def _integrate(self, cloud: CloudSample, pose: PoseSample):
        source = cloud.frame_id.lstrip('/')
        if self.initial_sensor_frame is None:
            self.initial_sensor_frame = source
        if source != self.initial_sensor_frame or (self.cfg.sensor_frame_id and source != self.cfg.sensor_frame_id):
            self._reject('SENSOR_FRAME_CHANGED')
            return
        if source == self.frame_id:
            # Explicitly already in the odometry/global frame; no extra transform.
            sensor_origin = pose.position  # approximate for projection; never for navigation
            converted = cloud.xyz
        elif source == self.child_frame_id or self.cfg.sensor_to_body is not None:
            # Raw sensor-frame points need calibrated sensor-to-body extrinsics.
            if source != self.child_frame_id and self.cfg.sensor_to_body is None:
                self._reject('MISSING_EXTRINSIC')
                return
            extrinsic = self.cfg.sensor_to_body
            if source == self.child_frame_id:
                # Exact body frame: identity (unless a configured sensor extrinsic is
                # supplied for a sensor with a distinct frame, handled below).
                origin_body = (0., 0., 0.)
                body_points = cloud.xyz
            else:
                origin_body = extrinsic.translation
                body_points = tuple(_add(_quat_rotate(extrinsic.rotation_xyzw,p), origin_body)
                                    for p in cloud.xyz)
            sensor_origin = _add(pose.position, _quat_rotate(pose.quaternion_xyzw,origin_body))
            converted = tuple(_add(pose.position,_quat_rotate(pose.quaternion_xyzw,p)) for p in body_points)
        else:
            self._reject('MISSING_EXTRINSIC')
            return
        # This is a diagnostic XY slice; z filtering intentionally avoids many floor returns.
        selected = 0
        for world in converted:
            relative_height = world[2]-pose.position[2]
            if not self.cfg.min_relative_z_m <= relative_height <= self.cfg.max_relative_z_m:
                continue
            d = sqrt(sum((p-o)**2 for p,o in zip(world,sensor_origin)))
            if d < self.cfg.min_range_m or d < 1e-6:
                continue
            if d <= self.cfg.max_range_m:
                self.grid.ray(sensor_origin,world,True)
            else:
                u = self.cfg.max_range_m/d
                endpoint = tuple(o+u*(p-o) for p,o in zip(world,sensor_origin))
                self.grid.ray(sensor_origin,endpoint,False)
            selected += 1
            if selected >= self.cfg.max_points_per_cloud:
                break
        self.accepted_clouds += 1
        self.map_version += 1
        self.state_seq += 1
        self.last_processed_stamp_s = cloud.stamp_s

    def health(self, now_s: float | None = None) -> dict:
        if not self.poses:
            status = 'NO_ODOMETRY'
            age = None
        else:
            age = max(0., (now_s if now_s is not None else self.poses[-1].stamp_s)-self.poses[-1].stamp_s)
            status = 'STALE_ODOMETRY' if age > self.cfg.max_odom_age_s else ('NO_REGISTERED_SCANS' if not self.accepted_clouds else 'OBSERVATION_ONLY')
        return {
            'status':status, 'localization_age_s':age,
            'last_odom_stamp_s':self.poses[-1].stamp_s if self.poses else None,
            'last_scan_stamp_s':self.last_processed_stamp_s,
            'accepted_clouds':self.accepted_clouds, 'dropped_clouds':self.dropped_clouds,
            'pending_clouds':len(self.pending), 'drop_reasons':dict(self.rejection_reasons),
            'camera_frames':self.camera_frames,'last_camera_stamp_s':self.last_camera_stamp_s,
            'known_debug_cells':self.grid.known_cells(),
            'resources':'UNAVAILABLE', 'perception':'UNAVAILABLE',
            'planning_actions':'DISABLED',
            'debug_projection_safe_for_navigation':False,
        }

    def snapshot(self, now_s: float | None = None) -> MissionSnapshot:
        if not self.poses:
            raise RuntimeError('No valid odometry yet')
        ts = max(self.poses[-1].stamp_s,now_s or 0)
        pose = self.poses[-1]
        # Real readings cannot be used to claim mission completion without
        # registered semantic verifiers/executor telemetry. Leave all tasks open.
        states = [GoalProgress(goal_id=g.id, status=GoalStatus.UNKNOWN if g.depends_on else GoalStatus.PENDING)
                  for g in self.spec.goals]
        snap = MissionSnapshot(
            schema_version='2.0', mission_id=self.spec.mission_id,
            spec_revision=self.spec.revision, state_seq=self.state_seq,
            map_version=self.map_version,
            captured_at=datetime.fromtimestamp(ts,timezone.utc),
            robot_pose=Position(x=float(pose.position[0]), y=float(pose.position[1]),
                                z=float(pose.position[2]), frame_id=self.frame_id),
            goals=states, bindings={}, entities=[], frontiers=[], viewpoint_ids=[],region_ids=[],
            observations=[], resources=Resources(
                remaining_energy_wh=0.0,reserve_energy_wh=0.0,
                remaining_flight_time_s=0.0,reserve_time_s=0.0,
                current_return_energy_wh=0.0,current_return_time_s=0.0),
            candidates=[])
        validate_spec_snapshot(self.spec,snap)
        return snap

    def visual_frame(self, now_s: float | None = None) -> VisualizationFrame:
        snap = self.snapshot(now_s)
        frame = frame_from_observed(
            snap, self.grid.as_tuple(),
            home=VisualPoint(x=self.home[0],y=self.home[1]),
            executed_path=self.path,
            cell_size_m=self.cfg.resolution_m)
        return frame.model_copy(update={'origin_x_m':self.cfg.origin_x_m,
                                        'origin_y_m':self.cfg.origin_y_m,
                                        'planner':VisualPlanner(
            mode='ROS_BRIDGE',status=self.health(now_s)['status'],
            rationale='Diagnostic 2D projection only; actions disabled')})
