"""Synthetic ROS-like event fixture; NO real ROS, FAST-LIVO2, LiDAR or flight data."""
from __future__ import annotations
import json
from pathlib import Path
from .core import PoseSample,CloudSample


def write_fixture(path: str | Path, *, start_s:float=1793620000.0):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Two parallel vertical walls on either side of a horizontal route.
    with path.open('w') as f:
        for i in range(41):
            t = start_s+i*.1
            px = i*.15
            pose = {'type':'odom','stamp_s':t,'frame_id':'map','child_frame_id':'base_link',
                    'position':[px,0.,1.5],'quaternion_xyzw':[0.,0.,0.,1.]}
            f.write(json.dumps(pose)+'\n')
            if i % 2 == 0:
                world = [(x,y,1.5) for x in (px+1.,px+2.,px+3.) for y in (-2.,2.)]
                local = [[round(x-px,5),y,z-1.5] for x,y,z in world]
                cloud = {'type':'cloud','stamp_s':t,'frame_id':'base_link','xyz':local}
                f.write(json.dumps(cloud)+'\n')
    return path


def iter_fixture(path: str|Path):
    with Path(path).open() as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            if obj['type']=='odom':
                yield ('odom',PoseSample(obj['stamp_s'],obj['frame_id'],obj['child_frame_id'],
                                         tuple(obj['position']),tuple(obj['quaternion_xyzw'])))
            elif obj['type']=='cloud':
                yield ('cloud',CloudSample(obj['stamp_s'],obj['frame_id'],
                                           tuple(tuple(p) for p in obj['xyz'])))
            else:
                raise ValueError('Unknown fixture record type')
