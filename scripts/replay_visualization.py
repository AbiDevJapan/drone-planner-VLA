"""Replay observation-only visualization logs, without running the simulator."""
from __future__ import annotations
import argparse
from pathlib import Path
from visualization.frame import VisualizationFrame
from visualization.sinks import FrameSinkGroup, GifFrameRecorder, LiveMatplotlibSink
from visualization.matplotlib_view import DashboardRenderer


def replay(log: Path, *, live=False, delay=0.35, gif: Path | None = None,
           png: Path | None = None, fps: float = 2.0):
    sinks=[]
    if live:
        sinks.append(LiveMatplotlibSink(delay))
    if gif:
        sinks.append(GifFrameRecorder(gif, fps))
    fanout=FrameSinkGroup(*sinks)
    last=None
    count=0
    try:
        with log.open(encoding='utf8') as f:
            for line in f:
                if not line.strip():
                    continue
                frame = VisualizationFrame.model_validate_json(line)
                fanout.on_frame(frame)
                last=frame
                count+=1
    finally:
        fanout.close()
    if png and last is not None:
        renderer=DashboardRenderer()
        try:
            renderer.save_png(last,png)
        finally:
            renderer.close()
    return count


def main():
    parser=argparse.ArgumentParser(description='Replay a Step 2 visualization JSONL trace')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--delay', type=float, default=0.35)
    parser.add_argument('--gif', type=Path)
    parser.add_argument('--png', type=Path)
    parser.add_argument('--fps', type=float, default=2.0)
    args=parser.parse_args()
    n=replay(args.input,live=args.live,delay=args.delay,gif=args.gif,png=args.png,fps=args.fps)
    print(f'Replayed {n} visualization frames')


if __name__ == '__main__':
    main()
