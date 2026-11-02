# Mission-Adaptive UAV — Step 5: FAST-LIVO2 / ROS 2 observation bridge

**Purpose:** Connect REAL FAST-LIVO2 odometry and Livox LiDAR input to the same generalized **MissionSnapshot v2** and Step 2–4 visualization interfaces, starting with ROS bag playback. The full Step 4 simulator, Ollama planner, safety supervisor, scenarios and tests remain in this package, unchanged in behavior.

**This is an observation-only integration milestone, NOT autonomous real-drone control.** It cannot mark fire extinguishers detected from point clouds, infer verified inspection results, generate safe 3D exploration frontiers, estimate actual battery energy, or command the flight controller. All `MissionSnapshot.candidates` and `.frontiers` are empty; telemetry resources are set to zero and explicitly labeled **UNAVAILABLE** in `bridge_health.jsonl` to prevent simulated numbers from being used as real aircraft resources. Semantic evidence and task completion require trusted separate detectors/verifiers. Step 6 will add proper occupancy/frontiers/telemetry and action candidate generation.

## What is included

- `ros2_bridge/core.py`: **ROS-independent** pose and sensor synchronization, configurable map frames, bounded buffers, LiDAR-to-body extrinsics, quaternion transforms, 2D diagnostic projection, staleness and rejection counters, and `MissionSnapshot v2` creation.
- `ros2_bridge/ros_adapters.py`: ROS 2 Humble adapters for `/aft_mapped_to_init` (`nav_msgs/msg/Odometry`), `/livox/lidar` (`livox_ros_driver2/msg/CustomMsg` OR `sensor_msgs/msg/PointCloud2`), optional `/left_camera/image` (`sensor_msgs/msg/Image` metadata only). Supports direct `rosbag2_py` reading or a live ROS 2 node.
- `ros2_bridge/fixture.py`: synthetic ROS-like JSONL events to test on macOS **without ROS**.
- `scripts/run_step5.py`: single CLI for `--source fixture`, `--source bag`, `--source live`.
- Existing `visualization/`: ROS-independent dashboard and replayable frame JSONL. Visual frames now have grid world origin offsets, so map and drone coordinates coincide.
- **92 passing tests in this build**, including all Step 4 regressions and new synchronization/validation tests.

## A. First run on MacBook (no ROS installation)

```bash
cd mission_adaptive_uav_step5
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz]'

python -m scripts.run_step5 --source fixture \
  --output-dir outputs/step5_fixture \
  --viz-png outputs/step5_fixture/final_dashboard.png \
  --viz-log outputs/step5_fixture/frames.jsonl \
  --viz-gif outputs/step5_fixture/demo.gif

python -m pytest -q
```

The fixture is **synthetic and NOT data from your UAV**. The expected test run has 21 accepted synthetic clouds, nonzero observed debug cells, and zero confirmed mission goals. The dashboard shows a simulated traverse between two wall lines. An example output PNG and GIF are bundled under `outputs/step5_fixture/`.

The previous Step 4 LLM demo continues to work:

```bash
python -m scripts.run_step4 --planner scripted --mission warehouse --seed 0
python -m scripts.run_step4 --check-ollama
```

**Do not use `--source fixture` to claim real ROS integration success.** The actual ROS 2 APIs have not been executed in the development container (ROS 2 Humble is not installed there).

## B. Run on Ubuntu 22.04 + ROS 2 Humble (Jetson/workstation)

Activate your ROS 2 system and existing FAST-LIVO2 workspace overlay. Since ROS 2 Humble's `rclpy` and `rosbag2_py` depend on Ubuntu's native Python 3.10, use its Python and a **system-site-packages** virtual environment instead of your macOS Python environment.

```bash
source /opt/ros/humble/setup.bash
source ~/fast_ws/install/setup.bash  # adjust to your actual workspace

cd mission_adaptive_uav_step5
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viz]'
python -c 'import rclpy, rosbag2_py; from livox_ros_driver2.msg import CustomMsg; print("ROS bridge dependencies OK")'
```

If ROS packages are missing, install their Ubuntu/ROS 2 apt packages, or source the Livox overlay containing the generated `livox_ros_driver2` Python messages. Do not attempt to install `rclpy` or Livox ROS message definitions using ordinary `pip install rclpy` in the macOS environment.

Before replay, inspect bag topics and odometry frames:

```bash
ros2 bag info /path/to/your_bag
ros2 topic list -t
ros2 topic info /livox/lidar --verbose
ros2 topic echo /aft_mapped_to_init --once --field header
ros2 topic echo /aft_mapped_to_init --once --field child_frame_id
ros2 topic echo /livox/lidar --once --field header
```

The `ros2 topic` commands require topics to be active; if you are inspecting a recorded bag, first play it in another terminal (or use a small `ros2 bag play` segment). If `/livox/lidar` has multiple different message types on the same name, **identify the actual type in the recorded bag** and avoid ambiguous simultaneous publishers. `--cloud-type` selects the type for *live* subscriptions; the direct bag reader discovers the type from metadata.

### Option 1: read a recorded bag directly (recommended first)

```bash
python -m scripts.run_step5 --source bag \
  --bag /path/to/your_bag \
  --odom-topic /aft_mapped_to_init \
  --cloud-topic /livox/lidar \
  --camera-topic /left_camera/image \
  --output-dir outputs/real_bag_01 \
  --emit-every-clouds 50 \
  --viz-png outputs/real_bag_01/final_dashboard.png \
  --viz-log outputs/real_bag_01/frames.jsonl
```

Omit `--camera-topic` if that image topic isn't in the bag. Add `--storage-id mcap` if the bag uses MCAP instead of SQLite3. No live LiDAR or ROS bag playback process is required for this direct reader.

**Sensor frame warning:** FAST-LIVO2 odometry normally describes a body pose in an odometry/world frame, while raw Livox points are in a LiDAR sensor frame. The bridge **will reject raw Livox clouds** if no known LiDAR-to-body transform is configured. This is intentional: silently treating the LiDAR frame as the odometry frame would misregister the scan. Inspect `final_health.json` for `drop_reasons.MISSING_EXTRINSIC`.

Once you know your actual calibrated static transform `T_body_lidar` and point-cloud `header.frame_id`, add:

```bash
# Append to the preceding command (replace the measured values with YOUR calibration):
--sensor-frame ACTUAL_LIDAR_FRAME \
--sensor-to-body "TX,TY,TZ,QX,QY,QZ,QW"
```

`TX, TY, TZ` are measured **metres** from body origin to LiDAR origin, expressed in body coordinates; quaternion `QX,QY,QZ,QW` rotates the LiDAR frame into the body frame. Do **not** use guessed values for real mapping. If the cloud is already expressed in exactly the odometry's `frame_id`, no static transform is needed. If `child_frame_id` is missing in the odometry, add `--body-frame ACTUAL_BODY_FRAME` after checking your SLAM frame convention. Vary `--origin-x`, `--origin-y`, `--resolution` and grid dimensions for the extent of the bag.

The timestamp-matching gate defaults to **0.5 seconds maximum surrounding odometry gap**. Late scans are buffered until a future odometry sample arrives. Scans outside valid pose brackets or with frame mismatches are dropped with reasons recorded. Step 5 does NOT perform Livox per-point motion deskew or camera-LiDAR spatial calibration; it provides a limited visualization, not production occupancy mapping.

### Option 2: subscribe to live topics / bag playback

In a terminal where FAST-LIVO2 and Livox publishers are already running:

```bash
python -m scripts.run_step5 --source live \
  --odom-topic /aft_mapped_to_init \
  --cloud-topic /livox/lidar \
  --cloud-type custom \
  --camera-topic /left_camera/image \
  --output-dir outputs/live_bridge \
  --sensor-frame ACTUAL_LIDAR_FRAME \
  --sensor-to-body "TX,TY,TZ,QX,QY,QZ,QW"
```

Omit the camera option if the camera isn't publishing. Use `--cloud-type pointcloud2` for a standard `sensor_msgs/msg/PointCloud2` publisher. For *live playback* from a recorded bag, run `ros2 bag play /path/to/bag --clock` in another terminal and add `--use-sim-time` to the bridge command. Ensure sensor Header stamps and ROS clock are consistent; don't mix wall time and bag time. Press Ctrl+C to stop; logs are flushed. Prefer `--source bag` for initial reproducible debugging.

The bridge publishes **JSON strings**, not motor control commands:

- `/mission_bridge/snapshot` (`std_msgs/msg/String`): generalized `MissionSnapshot v2`, with real odometry pose but **empty** frontiers, entities, observations, and candidates.
- `/mission_bridge/health` (`std_msgs/msg/String`): stale/missing odometry, dropped scans, frame issues, camera frame count, unavailable resource flags.
- `/mission_bridge/visualization_frame` (`std_msgs/msg/String`): observed **diagnostic** 2D XY projection for local visualization. High-bandwidth locally; do NOT forward this entire grid to the offboard LLM.

To inspect:

```bash
ros2 topic echo /mission_bridge/health
ros2 topic hz /mission_bridge/snapshot
ros2 topic echo /mission_bridge/snapshot --once
```

## C. Outputs and acceptance criteria

| File | Contains |
|---|---|
| `snapshots.jsonl` | Valid v2 mission-state messages from real odometry, **no fabricated mission evidence/actions** |
| `bridge_health.jsonl` | Per-snapshot bridge state, stale timestamps, drop reasons, scan/camera counters |
| `final_snapshot.json` | Last validated mission snapshot |
| `final_health.json` | Final input and health diagnostics |
| `frames.jsonl` | Optional local visualization/replay frames |
| `final_dashboard.png` | Optional final diagnostic 2D projection |

Accept Step 5 when: (1) a real ROS bag or live stream creates valid odometry snapshots with the expected frame ID and coordinates, (2) LiDAR scans are timestamp-aligned and projected only with justified transforms, (3) missing/invalid localization appears in diagnostics, (4) the dashboard shows the actual pose trace and sensor observations, (5) disconnecting offboard Ollama doesn't affect the ROS bridge, and (6) no real action is executed or goal marked verified by the bridge alone. After that, Step 6 will build a **real, collision-aware 3D occupancy/frontier manager and flight-resource monitor**, using the same mission-state protocol.

## Limits / verification status

- The ROS-independent bridge and its synthetic replay path have been tested locally. **Native `rclpy`, `rosbag2_py`, and actual FAST-LIVO2 / Livox messages still require testing on your ROS 2 machine**. No actual user ROS bag was provided with this turn.
- The 2D XY projection is deliberately labeled **NOT NAVIGABLE**. It is not 3D collision checking, safe free-space evidence, a SLAM replacement, or a frontier source. Cells labeled 'free' are only free in that simplified ray projection, not flight-certified 3D space.
- Physical drone testing also needs an independent flight-controller failsafe, hardware return/land behavior, calibrated battery model, correct TF/extrinsics, and localization-failure handling.
- The camera adapter stores **only metadata** (frame/timestamps, dimensions, encoding); it does not classify an object, track images, or claim inspection confidence. The offboard Ollama remains disconnected from these sensors in Step 5 by design.
- The same dashboard/mission schemas are used; the existing simulation and Step 4 demonstration are retained unchanged.
