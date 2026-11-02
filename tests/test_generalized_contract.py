from copy import deepcopy
from pathlib import Path
import pytest
from pydantic import ValidationError
from interfaces import MissionSpec, MissionSnapshot, PlannerDecision, validate_spec_snapshot, validate_decision
from interfaces.validation import InvalidMissionState, StaleDecisionError, UnknownActionError

ROOT = Path(__file__).resolve().parents[1] / 'examples'
def load(name):
    spec = MissionSpec.model_validate_json((ROOT / f'{name}_spec.json').read_text())
    state = MissionSnapshot.model_validate_json((ROOT / f'{name}_state.json').read_text())
    return spec,state

def modified(spec, state):
    return deepcopy(spec.model_dump(mode='json')),deepcopy(state.model_dump(mode='json'))

def decision_from(state, action_id):
    return PlannerDecision(schema_version='2.0',mission_id=state.mission_id,spec_revision=state.spec_revision,state_seq=state.state_seq,map_version=state.map_version,action_id=action_id,rationale='Test decision')

@pytest.mark.parametrize('name',['warehouse','exit_sign','thermal_inspection'])
def test_three_mission_categories_validate(name):
    spec, state = load(name)
    validate_spec_snapshot(spec,state)

def test_decision_resolves_advertised_action():
    spec,state=load('exit_sign')
    assert validate_decision(spec,state,decision_from(state,'read_sign')).capability_id=='uav.inspect'

def test_wrong_observation_property_cannot_verify():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['observations'][1]['property']='color'
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_wrong_observation_value_cannot_verify():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['observations'][1]['value']='UNKNOWN'
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_confidence_threshold_cannot_be_bypassed():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['observations'][1]['confidence']=0.2
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_allowed_producer_is_required():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['observations'][1]['producer_id']='unknown_agent'
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_llm_cannot_be_evidence_producer_kind():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['observations'][1]['producer_kind']='LLM'
    with pytest.raises(ValidationError): MissionSnapshot.model_validate(b)

def test_binding_points_to_actual_verified_entity():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['entities'].append({'id':'ext_3','class_id':'fire_extinguisher','confidence':0.9,'position':None})
    b['bindings']['target_equipment']='ext_3'
    with pytest.raises(InvalidMissionState,match='Bound entity'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_cannot_mix_evidence_from_two_objects():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['entities'].append({'id':'ext_3','class_id':'fire_extinguisher','confidence':0.9,'position':None})
    b['observations'][1]['subject_id']='ext_3'
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_expired_evidence_cannot_verify():
    spec,state=load('thermal_inspection'); a,b=modified(spec,state)
    b['observations'][0]['observed_at']='2026-11-02T03:00:00Z'
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_numeric_comparison_not_bool():
    spec,state=load('thermal_inspection'); a,b=modified(spec,state)
    b['observations'][0]['value']=True
    with pytest.raises(InvalidMissionState,match='insufficient matching evidence'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_target_referential_integrity():
    spec,state=load('exit_sign'); a,b=modified(spec,state)
    b['candidates'][0]['target']['ref']='other_sign'
    with pytest.raises(InvalidMissionState,match='unknown target'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_capability_target_kind_not_arbitrary():
    spec,state=load('exit_sign'); a,b=modified(spec,state)
    b['candidates'][0]['target']={'kind':'GOAL','ref':'read'}
    with pytest.raises(InvalidMissionState,match='wrong target kind'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_unknown_goal_and_missing_dependency():
    spec,state=load('warehouse'); a,b=modified(spec,state)
    b['goals'][0]['status']='PENDING'; b['goals'][0]['evidence_ids']=[]
    b['bindings']={}
    with pytest.raises(InvalidMissionState,match='dependencies not verified'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))

def test_lookahead_time_and_energy_are_typed():
    _,state=load('exit_sign'); b=deepcopy(state.model_dump(mode='json'))
    b['candidates'][0]['cost']['execute_time_s']='a while'
    with pytest.raises(ValidationError): MissionSnapshot.model_validate(b)

def test_stale_seq_rejected():
    spec,state=load('exit_sign')
    d=decision_from(state,'read_sign').model_copy(update={'state_seq':state.state_seq-1})
    with pytest.raises(StaleDecisionError):validate_decision(spec,state,d)

def test_stale_spec_revision_rejected():
    spec,state=load('exit_sign')
    d=decision_from(state,'read_sign').model_copy(update={'spec_revision':state.spec_revision+1})
    with pytest.raises(StaleDecisionError):validate_decision(spec,state,d)

def test_unknown_action_rejected():
    spec,state=load('exit_sign')
    d=decision_from(state,'improvise')
    with pytest.raises(UnknownActionError):validate_decision(spec,state,d)

def test_cyclic_goal_graph_rejected():
    spec,_=load('warehouse'); a=deepcopy(spec.model_dump(mode='json'))
    a['goals'][0]['depends_on']=['access']
    with pytest.raises(ValidationError,match='Cyclic'):
        MissionSpec.model_validate(a)

def test_missing_evidence_for_verified_goal_rejected():
    _,state=load('warehouse'); b=deepcopy(state.model_dump(mode='json'))
    b['goals'][1]['evidence_ids']=[]
    with pytest.raises(ValidationError,match='VERIFIED goal requires evidence'):
        MissionSnapshot.model_validate(b)

def test_unknown_spec_revision_rejected():
    spec,state=load('warehouse'); b=deepcopy(state.model_dump(mode='json'))
    b['spec_revision']=23
    with pytest.raises(InvalidMissionState,match='Mismatched specification'):
        validate_spec_snapshot(spec,MissionSnapshot.model_validate(b))
