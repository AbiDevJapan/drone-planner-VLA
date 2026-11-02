"""Replaceable visualization outputs: GUI, reproducible JSONL trace, animated GIF.

The engine stays unaware of these sinks. Later, a ROS2 publisher can consume
exactly the same `VisualizationFrame` events and publish OccupancyGrid/Markers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
from .frame import VisualizationFrame


class FrameSink(Protocol):
    def on_frame(self, frame: VisualizationFrame) -> None: ...
    def close(self) -> None: ...


class FrameSinkGroup:
    def __init__(self, *sinks: FrameSink):
        self.sinks = sinks

    def on_frame(self, frame: VisualizationFrame):
        for sink in self.sinks:
            sink.on_frame(frame)

    def close(self):
        for sink in self.sinks:
            sink.close()


class JsonlFrameRecorder:
    """Stream visualization snapshots to disk for replay/debugging.

    Frames contain the observed occupancy and may be much larger than the
    compact offboard planning messages. Use this for local logs only.
    """
    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.destination.open('w', encoding='utf8')

    def on_frame(self, frame: VisualizationFrame):
        self.file.write(frame.model_dump_json() + '\n')
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class GifFrameRecorder:
    """Small-simulation-only demo recorder (keeps rendered frames in memory)."""
    def __init__(self, destination: Path, fps: float = 2.0):
        from .matplotlib_view import DashboardRenderer
        if not 0 < fps <= 60:
            raise ValueError('GIF fps must be in (0, 60]')
        self.destination = Path(destination)
        self.duration_ms = max(20, round(1000 / fps))
        self.renderer = DashboardRenderer()
        self.images = []

    def on_frame(self, frame: VisualizationFrame):
        self.renderer.draw(frame)
        self.images.append(self.renderer.image())

    def close(self):
        if self.images:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.images[0].save(self.destination, save_all=True,
                                append_images=self.images[1:],
                                duration=self.duration_ms, loop=0,
                                optimize=True)
        self.images.clear()
        self.renderer.close()


class LiveMatplotlibSink:
    """Update a Matplotlib window after each high-level simulator action."""
    def __init__(self, delay_s: float = 0.3, *, hold_final: bool = True):
        import matplotlib.pyplot as plt
        from .matplotlib_view import DashboardRenderer, interactive_backend_available
        if not interactive_backend_available():
            raise RuntimeError('No interactive Matplotlib backend. Run from a desktop terminal '
                               'with Tk/Qt/macOS plotting available, or use --viz-gif '
                               'for a headless recording.')
        if delay_s < 0:
            raise ValueError('delay_s cannot be negative')
        self.renderer = DashboardRenderer()
        self.plt = plt
        self.delay_s = delay_s
        self.hold_final = hold_final
        plt.ion()
        plt.show(block=False)

    def on_frame(self, frame: VisualizationFrame):
        self.renderer.draw(frame)
        # Enter GUI event loop to keep the window responsive in the synchronous demo.
        self.plt.pause(max(0.001, self.delay_s))

    def close(self):
        if self.hold_final and self.plt.fignum_exists(self.renderer.fig.number):
            self.plt.ioff()
            self.plt.show(block=True)
        self.renderer.close()
