import json
import math
from pathlib import Path
import pytest
from interfaces import MissionSpec,validate_spec_snapshot
from ros2_bridge import BridgeConfig, SensorToBody, PoseSample, CloudSample, CameraSample, RosObservationBridge
from ros2_bridge.fixture import iter_fixture,write_fixture
from ros2_bridge.core import _quat_rotate

ROOT=Path(__file__).resolve().parents[1]
SPEC=MissionSpec.model_validate_json((ROOT/'examples/warehouse_spec.json').read_text())
T=1793620000.


def pose(t=T,x=0,frame='map',child='base_link'):
    return PoseSample(t,frame,child,(float(x),0.,1.5),(0.,0.,0.,1.))


def cloud(t=T,frame='base_link'):
    return CloudSample(t,frame,((2.,1.,0.),(3.,-1.,0.),(1.5,0.,-1.8)))


def test_schema_no_fake_semantics_and_no_actions():
    b=RosObservationBridge(SPEC)
    b.receive_pose(pose())
    b.receive_cloud(cloud())
    s=b.snapshot()
    assert s.candidates==s.frontiers==s.observations==[]
    assert s.entities==[] and s.bindings=={}
    assert all(g.status.value!='VERIFIED' for g in s.goals)
    assert s.resources.remaining_energy_wh==0
    assert b.health()['planning_actions']=='DISABLED'
    assert b.health()['debug_projection_safe_for_navigation'] is False
    validate_spec_snapshot(SPEC,s)


def test_odd_frame_rejected_and_no_safe_cloud():
    b=RosObservationBridge(SPEC)
    b.receive_pose(pose())
    b.receive_cloud(cloud(frame='livox_frame'))
    assert b.accepted_clouds==0 and b.dropped_clouds==1
    assert b.health()['drop_reasons']['MISSING_EXTRINSIC']==1
    assert b.grid.known_cells()==0


def test_explicit_extrinsic_and_world_frame():
    cfg=BridgeConfig(sensor_to_body=SensorToBody((.1,0.,0.),(0.,0.,0.,1.)),sensor_frame_id='livox_frame')
    b=RosObservationBridge(SPEC,cfg)
    b.receive_pose(pose())
    b.receive_cloud(cloud(frame='livox_frame'))
    assert b.accepted_clouds==1 and b.grid.known_cells()>0
    b2=RosObservationBridge(SPEC)
    b2.receive_pose(pose())
    b2.receive_cloud(cloud(frame='map'))
    assert b2.accepted_clouds==1


def test_interpolation_waits_and_gap_drop():
    b=RosObservationBridge(SPEC,BridgeConfig(max_interpolation_gap_s=.5))
    b.receive_pose(pose(T,0))
    b.receive_cloud(cloud(T+.2))
    assert b.accepted_clouds==0 and len(b.pending)==1
    b.receive_pose(pose(T+.4,4))
    assert b.accepted_clouds==1
    interp=b._pose_at(T+.2)
    assert interp.position[0]==pytest.approx(2)
    long=RosObservationBridge(SPEC,BridgeConfig(max_interpolation_gap_s=.5))
    long.receive_pose(pose(T))
    long.receive_cloud(cloud(T+.5))
    long.receive_pose(pose(T+1))
    assert long.dropped_clouds==1
    assert long.health()['drop_reasons']['POSE_GAP_TOO_LARGE']==1


def test_out_of_order_pose_and_frame_change():
    b=RosObservationBridge(SPEC)
    b.receive_pose(pose())
    with pytest.raises(ValueError,match='Out-of-order'):b.receive_pose(pose())
    with pytest.raises(ValueError,match='frame changed'):b.receive_pose(pose(T+.1,frame='odom'))


def test_stale_and_no_odom_fail_closed():
    b=RosObservationBridge(SPEC)
    with pytest.raises(RuntimeError,match='No valid odometry'):b.snapshot()
    assert b.health()['status']=='NO_ODOMETRY'
    b.receive_pose(pose())
    assert b.health(T+3)['status']=='STALE_ODOMETRY'


def test_bounded_pending_cloud_queue():
    b=RosObservationBridge(SPEC,BridgeConfig(max_pending_clouds=2))
    for i in range(5):b.receive_cloud(cloud(T+i*.1))
    assert len(b.pending)==2 and b.dropped_clouds==3


def test_fixture_replay_and_viz_origin(tmp_path):
    source=write_fixture(tmp_path/'ros.jsonl')
    b=RosObservationBridge(SPEC)
    for kind,item in iter_fixture(source):
        if kind=='odom':b.receive_pose(item)
        else:b.receive_cloud(item)
    b.flush_end_of_stream()
    assert b.accepted_clouds==21
    assert b.grid.known_cells()>20
    assert b.snapshot().robot_pose.x==pytest.approx(6.)
    frame=b.visual_frame()
    assert frame.origin_x_m==-25 and frame.origin_y_m==-25
    assert frame.drone.x==6
    assert frame.safety is None
    assert frame.planner.mode=='ROS_BRIDGE'


def test_quat_rotation():
    assert _quat_rotate((0.,0.,math.sqrt(.5),math.sqrt(.5)),(1.,0.,0.))==pytest.approx((0.,1.,0.))


def test_bad_quaternion_and_nan_rejected():
    with pytest.raises(ValueError): PoseSample(T,'map','base_link',(0,0,0),(0,0,0,0))
    with pytest.raises(ValueError): CloudSample(T,'map',((float('nan'),2,3),))


def test_sensor_frame_changed_even_with_extrinsic():
    cfg=BridgeConfig(sensor_to_body=SensorToBody((0,0,0),(0,0,0,1)), sensor_frame_id='livox_frame')
    b=RosObservationBridge(SPEC,cfg)
    b.receive_pose(pose())
    b.receive_cloud(cloud(frame='livox_frame'))
    b.receive_cloud(cloud(frame='other_sensor'))
    assert b.accepted_clouds==1 and b.dropped_clouds==1


def test_pending_scan_not_extrapolated_end_of_stream():
    b=RosObservationBridge(SPEC)
    b.receive_pose(pose())
    b.receive_cloud(cloud(T+.1))
    b.flush_end_of_stream()
    assert b.dropped_clouds==1 and b.health()['pending_clouds']==0


def test_camera_only_metadata_does_not_verify_goals():
    b=RosObservationBridge(SPEC)
    b.receive_pose(pose())
    b.receive_camera(CameraSample(T+.1,'left_camera',1920,1080,'rgb8'))
    assert b.health()['camera_frames']==1
    assert b.snapshot().observations==[]
    assert not b.snapshot().candidates
    with pytest.raises(ValueError,match='Camera frame changed'):
        b.receive_camera(CameraSample(T+.2,'wrong_camera',1920,1080,'rgb8'))
