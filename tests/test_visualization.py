"""Visualization must be passive, reproducible, observation-only and replayable."""
from pathlib import Path
import pytest
from pydantic import ValidationError

from simulation import load_scenario, ExplorationSimulator
from scripts.run_step2 import run
from visualization import (
    VisualizationFrame, frame_from_simulator, FrameSinkGroup, JsonlFrameRecorder,
)
from visualization.adapters import frame_from_observed
from visualization.frame import VisualPoint


def make_sim(seed=0):
    return ExplorationSimulator(*load_scenario('warehouse'), seed=seed)


def test_initial_viz_frame_contains_only_observed_map_and_no_hidden_objects():
    sim=make_sim()
    frame=frame_from_simulator(sim)
    assert frame.schema_version == 'viz-1.0'
    assert frame.mission_id == sim.spec.mission_id
    assert frame.state_seq == 0
    assert frame.coverage == pytest.approx(sim.map.coverage())
    assert any(-1 in row for row in frame.occupancy)
    assert not frame.discovered_entities
    json_blob=frame.model_dump_json()
    assert 'ext_2' not in json_blob
    assert 'fire_extinguisher' not in json_blob
    assert 'access_state' not in json_blob
    assert 'BLOCKED' not in json_blob
    assert 'terrain' not in json_blob
    assert frame.drone == frame.home


def test_viewer_frame_is_a_copy_not_a_reference_into_private_simulator_state():
    sim=make_sim()
    old=frame_from_simulator(sim)
    sim.select_action(next(a.id for a in sim.snapshot().candidates
                           if a.capability_id == 'uav.explore'))
    new=frame_from_simulator(sim)
    assert old.map_version < new.map_version
    assert old.occupancy != new.occupancy
    assert old.occupancy == VisualizationFrame.model_validate_json(old.model_dump_json()).occupancy


def test_frame_rejects_invalid_grid_and_nonfinite_positions():
    frame=frame_from_simulator(make_sim())
    data=frame.model_dump()
    data['occupancy']=[[-1, 0], [0]]
    with pytest.raises(ValidationError, match='rectangular'):
        VisualizationFrame.model_validate(data)
    data=frame.model_dump()
    data['occupancy']=[[7]]
    with pytest.raises(ValidationError, match='Grid supports'):
        VisualizationFrame.model_validate(data)
    data=frame.model_dump()
    data['drone']={'x': float('nan'), 'y': 0}
    with pytest.raises(ValidationError, match='finite'):
        VisualizationFrame.model_validate(data)


def test_viz_publisher_does_not_change_action_sequence_or_contract(tmp_path):
    sim_baseline, states_baseline=run('warehouse',seed=2,max_steps=40,verbose=False)
    log=tmp_path/'event_log.jsonl'
    recorder=JsonlFrameRecorder(log)
    try:
        sim_visual, states_visual=run('warehouse',seed=2,max_steps=40,verbose=False,
                                      frame_sink=recorder)
    finally:
        recorder.close()
    assert [s.model_dump_json() for s in states_baseline] == [s.model_dump_json() for s in states_visual]
    assert sim_baseline.action_history == sim_visual.action_history
    raw=log.read_text().strip().splitlines()
    assert len(raw) == len(states_visual)
    frames=[VisualizationFrame.model_validate_json(line) for line in raw]
    assert frames[0].selected_action_id is None
    assert [f.state_seq for f in frames] == list(range(len(frames)))
    assert [f.selected_action_id for f in frames[1:]] == sim_visual.action_history
    assert frames[-1].goals['return'] == 'VERIFIED'


def test_frame_from_observed_can_be_used_without_simulator():
    sim=make_sim()
    snapshot=sim.snapshot()
    frame=frame_from_observed(snapshot, sim.map.known,
                              home=VisualPoint(x=1.0,y=2.0))
    assert frame.home.x == 1.0
    assert frame.map_version == snapshot.map_version


def test_replay_and_gif_optional(tmp_path):
    pytest.importorskip('matplotlib')
    pytest.importorskip('PIL')
    from visualization import GifFrameRecorder
    from scripts.replay_visualization import replay
    from PIL import Image
    json_log=tmp_path/'events.jsonl'
    output_gif=tmp_path/'first.gif'
    output_png=tmp_path/'last.png'
    sinks=FrameSinkGroup(JsonlFrameRecorder(json_log), GifFrameRecorder(output_gif, fps=3))
    try:
        _, states=run('mapping',seed=0,max_steps=40,verbose=False,frame_sink=sinks)
    finally:
        sinks.close()
    assert output_gif.exists() and output_gif.stat().st_size > 10000
    with Image.open(output_gif) as img:
        assert getattr(img,'n_frames',1) >= 2
    replay_gif=tmp_path/'replay.gif'
    n=replay(json_log,gif=replay_gif,png=output_png,fps=3)
    assert n == len(states)
    assert replay_gif.exists() and output_png.exists()
