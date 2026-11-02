"""Step 5: ROS2 observation bridge. Fixture can run on macOS without ROS."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from interfaces import MissionSpec
from ros2_bridge import BridgeConfig, SensorToBody, RosObservationBridge
from ros2_bridge.fixture import iter_fixture,write_fixture


def build_parser():
    p=argparse.ArgumentParser(description='ROS2 FAST-LIVO2 read-only observation bridge')
    p.add_argument('--source',choices=['fixture','bag','live'],default='fixture')
    p.add_argument('--spec',default='examples/warehouse_spec.json')
    p.add_argument('--fixture',default='examples/ros_events.jsonl')
    p.add_argument('--bag',help='ROS 2 bag folder containing metadata.yaml')
    p.add_argument('--storage-id',default='sqlite3',choices=['sqlite3','mcap'])
    p.add_argument('--odom-topic',default='/aft_mapped_to_init')
    p.add_argument('--cloud-topic',default='/livox/lidar')
    p.add_argument('--camera-topic',help='Optional RGB topic, e.g. /left_camera/image; metadata only')
    p.add_argument('--cloud-type',choices=['custom','pointcloud2'],default='custom')
    p.add_argument('--body-frame',help='Odometry child frame if child_frame_id is empty')
    p.add_argument('--sensor-frame',help='Reject clouds from unexpected frame')
    p.add_argument('--sensor-to-body',help='Measured T_body_lidar: x,y,z,qx,qy,qz,qw (metres, ROS xyzw)')
    p.add_argument('--grid-width',type=int,default=100)
    p.add_argument('--grid-height',type=int,default=100)
    p.add_argument('--resolution',type=float,default=.5)
    p.add_argument('--origin-x',type=float,default=-25.)
    p.add_argument('--origin-y',type=float,default=-25.)
    p.add_argument('--max-interp-gap',type=float,default=.5)
    p.add_argument('--max-odom-age',type=float,default=1.)
    p.add_argument('--max-points',type=int,default=512)
    p.add_argument('--output-dir',default='outputs/step5_fixture')
    p.add_argument('--emit-every-clouds',type=int,default=3)
    p.add_argument('--max-messages',type=int,default=0)
    p.add_argument('--viz-log',help='Frame JSONL for replay')
    p.add_argument('--viz-gif',help='Animated output (not recommended for full bags)')
    p.add_argument('--viz-png',help='Final diagnostic dashboard PNG')
    p.add_argument('--live',action='store_true',help='Matplotlib debug viewer')
    p.add_argument('--frame-delay',type=float,default=.05)
    p.add_argument('--publish-period',type=float,default=.5)
    p.add_argument('--use-sim-time',action='store_true',help='When ros2 bag play --clock drives live topic replay')
    return p


def main(argv=None):
    p=build_parser()
    a=p.parse_args(argv)
    if a.emit_every_clouds < 1 or a.max_messages < 0:
        p.error('--emit-every-clouds must be >=1 and --max-messages >=0')
    if a.source=='bag' and not a.bag: p.error('--source bag requires --bag')
    extrinsic=None
    if a.sensor_to_body:
        try:
            parts=[float(v) for v in a.sensor_to_body.split(',')]
            if len(parts)!=7: raise ValueError('expected exactly 7 values')
            extrinsic=SensorToBody(tuple(parts[:3]),tuple(parts[3:]))
        except ValueError as e: p.error(f'Invalid --sensor-to-body: {e}')
    try:
        cfg=BridgeConfig(width=a.grid_width,height=a.grid_height,resolution_m=a.resolution,
                         origin_x_m=a.origin_x,origin_y_m=a.origin_y,
                         max_interpolation_gap_s=a.max_interp_gap,
                         max_odom_age_s=a.max_odom_age,max_points_per_cloud=a.max_points,
                         sensor_to_body=extrinsic,sensor_frame_id=a.sensor_frame)
    except ValueError as e: p.error(str(e))
    spec=MissionSpec.model_validate_json(Path(a.spec).read_text())
    bridge=RosObservationBridge(spec,cfg)
    from visualization.sinks import JsonlFrameRecorder, GifFrameRecorder, LiveMatplotlibSink, FrameSinkGroup
    sinks=[]
    if a.live: sinks.append(LiveMatplotlibSink(a.frame_delay,hold_final=False))
    if a.viz_log: sinks.append(JsonlFrameRecorder(Path(a.viz_log)))
    if a.viz_gif: sinks.append(GifFrameRecorder(Path(a.viz_gif)))
    sink=FrameSinkGroup(*sinks)
    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    events_written=0
    with (out/'snapshots.jsonl').open('w') as snapshots_file, \
         (out/'bridge_health.jsonl').open('w') as health_file:
        def emit(now_s=None):
            nonlocal events_written
            if not bridge.poses: return
            snap=bridge.snapshot(now_s)
            health=bridge.health(now_s)
            snapshots_file.write(snap.model_dump_json()+'\n')
            health_file.write(json.dumps(health)+'\n')
            snapshots_file.flush()
            health_file.flush()
            if sinks: sink.on_frame(bridge.visual_frame(now_s))
            events_written+=1

        try:
            if a.source=='live':
                from ros2_bridge.ros_adapters import run_live
                run_live(bridge,odom_topic=a.odom_topic,cloud_topic=a.cloud_topic,
                         cloud_type=a.cloud_type,body_frame=a.body_frame,
                         camera_topic=a.camera_topic,sink=None,
                         publish_period_s=a.publish_period,use_sim_time=a.use_sim_time,
                         record_callback=lambda s,h:emit())
            else:
                if a.source=='fixture':
                    path=Path(a.fixture)
                    if not path.exists():
                        write_fixture(path)
                    stream=iter_fixture(path)
                else:
                    from ros2_bridge.ros_adapters import iter_rosbag
                    stream=iter_rosbag(a.bag,odom_topic=a.odom_topic,
                                      cloud_topic=a.cloud_topic,storage_id=a.storage_id,
                                      body_frame=a.body_frame,camera_topic=a.camera_topic,
                                      max_points=a.max_points,
                                      max_messages=a.max_messages)
                previous=0
                for i,(kind,sample) in enumerate(stream):
                    if a.max_messages and i >= a.max_messages: break
                    if kind=='odom':bridge.receive_pose(sample)
                    elif kind=='cloud':bridge.receive_cloud(sample)
                    elif kind=='camera':bridge.receive_camera(sample)
                    if bridge.accepted_clouds != previous:
                        previous=bridge.accepted_clouds
                        if previous % a.emit_every_clouds == 0: emit()
                bridge.flush_end_of_stream()
                emit()
        finally:
            sink.close()
    h=bridge.health()
    print(json.dumps({'source':a.source,'accepted_clouds':h['accepted_clouds'],
          'dropped_clouds':h['dropped_clouds'],'health':h['status'],
          'known_debug_cells':h['known_debug_cells'],'snapshots_logged':events_written,
          'note':'Observation-only; no verified goals, action candidates, or flight control'},indent=2))
    if bridge.poses:
        final=bridge.snapshot()
        (out/'final_snapshot.json').write_text(final.model_dump_json(indent=2))
        (out/'final_health.json').write_text(json.dumps(h,indent=2))
        if a.viz_png:
            from visualization.matplotlib_view import DashboardRenderer
            r=DashboardRenderer()
            try:r.save_png(bridge.visual_frame(),Path(a.viz_png))
            finally:r.close()

if __name__=='__main__': main()
