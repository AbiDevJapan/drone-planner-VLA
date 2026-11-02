"""Validated scenario fixtures: ground truth is private to the simulator."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import Field, model_validator
from interfaces.common import Message, Id, NonNegative, Scalar
from interfaces import MissionSpec
from .grid import Cell, GridMap

class HiddenObject(Message):
    id: Id
    class_id: Id
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    properties: dict[Id, Scalar] = Field(default_factory=dict)
    producers: dict[Id, Id] = Field(default_factory=dict)
    placement_options: list[tuple[int, int]] = Field(default_factory=list)

    @property
    def cell(self) -> Cell:
        return (self.row, self.col)

class Scenario(Message):
    name: Id
    terrain: list[str]
    sensor_radius: int = Field(ge=1, le=30)
    detection_radius: NonNegative = 3.0
    inspection_radius: NonNegative = 2.0
    initial_energy_wh: NonNegative = 120.0
    initial_time_s: NonNegative = 1800.0
    reserve_energy_wh: NonNegative = 10.0
    reserve_time_s: NonNegative = 90.0
    altitude_m: float = 1.5
    objects: list[HiddenObject] = Field(default_factory=list)
    regions: list[Id] = Field(default_factory=lambda: ['survey_area'])
    coverage_producer: Id = 'sim_mapper'
    include_coverage: bool = False

    @model_validator(mode='after')
    def validate_world(self):
        if not self.terrain or len(set(map(len,self.terrain))) != 1:
            raise ValueError('terrain must be rectangular and nonempty')
        if len(set(o.id for o in self.objects)) != len(self.objects):
            raise ValueError('duplicate hidden object ID')
        if len(self.regions) != len(set(self.regions)):
            raise ValueError('duplicate region IDs')
        starts = [(r,c) for r,line in enumerate(self.terrain) for c,ch in enumerate(line) if ch == 'H']
        if len(starts) != 1:
            raise ValueError('exactly one H home required')
        for obj in self.objects:
            for row, col in [(obj.row, obj.col), *obj.placement_options]:
                if not (0 <= row < len(self.terrain) and 0 <= col < len(self.terrain[0])):
                    raise ValueError('object outside terrain')
                if self.terrain[row][col] == '#':
                    raise ValueError('object center cannot be inside a wall')
            if set(obj.properties) != set(obj.producers):
                raise ValueError('property and producer dictionaries must agree')
        return self

    @property
    def home(self) -> Cell:
        return next((r,c) for r,line in enumerate(self.terrain) for c,ch in enumerate(line) if ch == 'H')

    def create_map(self) -> GridMap:
        return GridMap(tuple(self.terrain), self.home, self.sensor_radius)


def load_scenario(name: str, directory: Path | None = None) -> tuple[MissionSpec, Scenario]:
    import json
    path = directory or (Path(__file__).resolve().parents[1] / 'examples')
    spec = MissionSpec.model_validate_json((path / f'{name}_spec.json').read_text())
    scenario = Scenario.model_validate_json((path / f'{name}_scenario.json').read_text())
    if scenario.name != name:
        raise ValueError('scenario name does not match filename')
    return spec, scenario
