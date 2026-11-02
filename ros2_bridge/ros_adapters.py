"""Imports ROS libraries only when a real ROS 2 stream/bag is used.

Humble's python/rclpy/rosbag2_py should come from /opt/ros/humble and the
Livox CustomMsg from a sourced livox_ros_driver2 overlay, NOT pip.
"""
from __future__ import annotations
import math
from .core import PoseSample, CloudSample, CameraSample


def stamp_s(header):
    value = float(header.stamp.sec) + 1e-9*float(header.stamp.nanosec)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('ROS Header has zero/invalid timestamp')
    return value


def odom_sample(msg, *, body_frame: str | None = None):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    child = msg.child_frame_id or body_frame
    if not child:
        raise ValueError('Odometry missing child_frame_id: pass --body-frame explicitly')
    return PoseSample(stamp_s(msg.header),msg.header.frame_id,child,
                      (float(p.x),float(p.y),float(p.z)),
                      (float(q.x),float(q.y),float(q.z),float(q.w)))


def camera_sample(msg):
    return CameraSample(stamp_s(msg.header),msg.header.frame_id,
                        int(msg.width),int(msg.height),str(msg.encoding))


def cloud_sample(msg, *, point_limit: int = 512):
    """Supports Livox CustomMsg or standard PointCloud2; downsample for debug.

    CustomMsg point.offset_time/timebase are NOT deskewed. Do not use this
    projection for navigation or scientifically accurate occupancy mapping.
    """
    if point_limit < 2:
        raise ValueError('point_limit must be >= 2')
    msg_type = type(msg).__name__
    if msg_type == 'CustomMsg':
        # Point number may differ from array length in corrupted recordings.
        pts = msg.points
        step = max(1, math.ceil(len(pts)/point_limit))
        out = [(float(p.x),float(p.y),float(p.z)) for p in pts[::step]]
    elif msg_type == 'PointCloud2':
        from sensor_msgs_py import point_cloud2
        # Humble versions may return structured numpy arrays or iterables.
        pts = point_cloud2.read_points(msg,field_names=('x','y','z'),skip_nans=True)
        n = len(pts) if hasattr(pts,'__len__') else None
        step = max(1, math.ceil(n/point_limit)) if n else 1
        out = []
        for i,p in enumerate(pts):
            if i % step:
                continue
            try:
                out.append((float(p['x']),float(p['y']),float(p['z'])))
            except (ValueError,TypeError,IndexError):
                out.append((float(p[0]),float(p[1]),float(p[2])))
            if len(out) >= point_limit:
                break
    else:
        raise TypeError(f'Unsupported cloud message {msg_type}')
    out = tuple(p for p in out if all(math.isfinite(v) for v in p))
    return CloudSample(stamp_s(msg.header),msg.header.frame_id,out)


def iter_rosbag(uri: str, *, odom_topic: str, cloud_topic: str,
                storage_id: str='sqlite3',body_frame: str | None=None,
                camera_topic: str | None=None,
                max_points: int=512, max_messages: int=0):
    """Yield ('odom' or 'cloud', sample) in recorded order from rosbag2_py."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri,storage_id=storage_id),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                            output_serialization_format='cdr'))
    types = {t.name:t.type for t in reader.get_all_topics_and_types()}
    selected = [odom_topic,cloud_topic]+([camera_topic] if camera_topic else [])
    for topic in selected:
        if topic not in types:
            raise ValueError(f'Bag has no topic {topic}. Available: {sorted(types)}')
    if types[odom_topic] != 'nav_msgs/msg/Odometry':
        raise ValueError(f'{odom_topic} must be nav_msgs/msg/Odometry, got {types[odom_topic]}')
    if types[cloud_topic] not in ('livox_ros_driver2/msg/CustomMsg','sensor_msgs/msg/PointCloud2'):
        raise ValueError(f'Unsupported cloud type: {types[cloud_topic]}')
    if camera_topic and types[camera_topic] != 'sensor_msgs/msg/Image':
        raise ValueError(f'Unexpected camera type: {types[camera_topic]}')
    msgs = {topic:get_message(types[topic]) for topic in selected}
    reader.set_filter(rosbag2_py.StorageFilter(topics=selected))
    count=0
    while reader.has_next():
        topic, raw, _bag_stamp = reader.read_next()
        msg = deserialize_message(raw,msgs[topic])
        if topic == odom_topic:
            yield ('odom',odom_sample(msg,body_frame=body_frame))
        elif topic == cloud_topic:
            yield ('cloud',cloud_sample(msg,point_limit=max_points))
        else:
            yield ('camera',camera_sample(msg))
        count += 1
        if max_messages and count >= max_messages:
            break


def run_live(core, *, odom_topic: str, cloud_topic: str, cloud_type: str,
             body_frame: str | None, camera_topic: str | None=None,
             sink=None, publish_period_s: float=0.5,
             use_sim_time: bool=False, record_callback=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    if cloud_type == 'custom':
        from livox_ros_driver2.msg import CustomMsg as CloudMsg
    elif cloud_type == 'pointcloud2':
        from sensor_msgs.msg import PointCloud2 as CloudMsg
    else:
        raise ValueError('Expected --cloud-type custom or pointcloud2')

    class BridgeNode(Node):
        def __init__(self):
            super().__init__('mission_ros_state_bridge')
            self.pub_snapshot = self.create_publisher(String,'/mission_bridge/snapshot',10)
            self.pub_health = self.create_publisher(String,'/mission_bridge/health',10)
            self.pub_viz = self.create_publisher(String,'/mission_bridge/visualization_frame',2)
            self.create_subscription(Odometry,odom_topic,self.on_pose,qos_profile_sensor_data)
            self.create_subscription(CloudMsg,cloud_topic,self.on_cloud,qos_profile_sensor_data)
            if camera_topic:
                self.create_subscription(Image,camera_topic,self.on_camera,qos_profile_sensor_data)
            self.create_timer(publish_period_s,self.on_timer)
            self.get_logger().info(f'Read-only ROS bridge: {odom_topic} + {cloud_topic} ({cloud_type}); no motor commands')

        def on_pose(self,msg):
            try: core.receive_pose(odom_sample(msg,body_frame=body_frame))
            except (ValueError, RuntimeError) as e: self.get_logger().warn(f'Rejected odometry: {e}')

        def on_cloud(self,msg):
            try: core.receive_cloud(cloud_sample(msg,point_limit=core.cfg.max_points_per_cloud))
            except (ValueError,RuntimeError,TypeError) as e: self.get_logger().warn(f'Rejected cloud: {e}')

        def on_camera(self,msg):
            try: core.receive_camera(camera_sample(msg))
            except (ValueError,RuntimeError) as e: self.get_logger().warn(f'Rejected camera metadata: {e}')

        def on_timer(self):
            now = self.get_clock().now().nanoseconds/1e9
            self.pub_health.publish(String(data=__import__('json').dumps(core.health(now))))
            if not core.poses: return
            snap = core.snapshot(now)
            frame = core.visual_frame(now)
            self.pub_snapshot.publish(String(data=snap.model_dump_json()))
            self.pub_viz.publish(String(data=frame.model_dump_json()))
            if sink: sink.on_frame(frame)
            if record_callback: record_callback(snap,core.health(now))

    rclpy.init(args=['--ros-args','-p','use_sim_time:=true'] if use_sim_time else None)
    node = BridgeNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
