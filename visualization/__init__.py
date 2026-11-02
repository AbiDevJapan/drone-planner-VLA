"""Optional, transport-independent UI frames and visualization sinks.

Only frame adapters know the simulator; renderers consume observation-only frames.
"""
from .frame import VisualizationFrame, VisualPoint, VisualEntity, VisualFrontier
from .adapters import frame_from_simulator
from .sinks import JsonlFrameRecorder, GifFrameRecorder, LiveMatplotlibSink, FrameSinkGroup

__all__ = [
    'VisualizationFrame', 'VisualPoint', 'VisualEntity', 'VisualFrontier',
    'frame_from_simulator', 'JsonlFrameRecorder', 'GifFrameRecorder',
    'LiveMatplotlibSink', 'FrameSinkGroup',
]
