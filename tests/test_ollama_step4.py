"""Step 4: deterministic fake Ollama and contract/safety adversarial tests.

No live Ollama service or GPU is required for CI.
"""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from pydantic import ValidationError

from offboard import OllamaPlanner, OllamaSettings, PlannerFailure, build_planning_input
from offboard.ollama_planner import HttpTransport
from simulation import ExplorationSimulator, load_scenario
from scripts.run_step4 import run


class FakeTransport:
    def __init__(self, force=None):
        self.force = force
        self.posts = []

    def post(self, url, payload, timeout_s):
        self.posts.append((url, payload, timeout_s))
        if self.force == 'timeout':
            raise PlannerFailure('simulated timeout')
        if self.force == 'malformed':
            return {'message': {'content': '{invalid json'}}
        if self.force == 'not_candidate':
            return {'message': {'content': '{"action_id":"invented_action","rationale":"bad"}'}}
        if self.force == 'extra_field':
            return {'message': {'content': '{"action_id":"return_home","rationale":"bad","verified":true}'}}
        if self.force == 'no_content':
            return {'message': {}}
        state = json.loads(payload['messages'][1]['content'])
        available = state['allowed_actions']
        actions = [a for cap in ('uav.inspect', 'uav.explore', 'uav.return_home')
                   for a in available if a['capability'] == cap]
        choice = actions[0]['id']
        return {'model': 'fake-ollama', 'message': {'content': json.dumps({
            'action_id': choice, 'rationale': 'Addresses unfinished mission objectives'})},
            'eval_count': 42, 'eval_duration': 250000000}

    def tags(self, url, timeout_s):
        return {'models': [{'name': 'qwen3:8b'}]}


def basic():
    spec, scenario = load_scenario('warehouse')
    sim = ExplorationSimulator(spec, scenario)
    return spec, sim, sim.snapshot(), sim.safety_assessments()


def settings():
    return OllamaSettings('http://127.0.0.1:11434', 'qwen3:8b')


def test_compact_prompt_exposes_only_observed_and_feasible_state():
    spec, sim, snap, assessed = basic()
    inp = build_planning_input(spec, snap, assessed)
    serialized = json.dumps(inp)
    assert inp['mission_id'] == snap.mission_id
    assert 'allowed_actions' in inp
    assert 'terrain' not in serialized
    assert 'occupancy' not in serialized
    assert 'path_history' not in serialized
    assert 'simulator_seed' not in serialized
    assert {a['id'] for a in inp['allowed_actions']} == sim.safety.safe_action_ids(snap, return_route_known=sim._known_return_route())
    for action in inp['allowed_actions']:
        assert 'energy_margin_wh' in action
        assert 'return_energy_wh_from_endpoint' in action


def test_schema_version_bound_action_selection():
    spec, sim, snap, assessed = basic()
    transport = FakeTransport()
    planner = OllamaPlanner(settings(), transport)
    result = planner.plan(spec, snap, assessed)
    assert result.decision.action_id in {a.id for a in snap.candidates}
    assert result.decision.state_seq == snap.state_seq
    assert result.decision.spec_revision == snap.spec_revision
    assert result.decision.map_version == snap.map_version
    assert result.eval_count == 42
    url, request, timeout_s = transport.posts[0]
    assert url.endswith('/api/chat')
    assert request['stream'] is False
    assert request['format']['properties']['action_id']['enum']
    assert request['options']['temperature'] == 0.1
    assert request['think'] is False
    assert request['model'] == 'qwen3:8b'


@pytest.mark.parametrize('failure', ['malformed', 'not_candidate', 'extra_field', 'no_content', 'timeout'])
def test_bad_response_never_selects_an_action(failure):
    spec, sim, snap, assessed = basic()
    with pytest.raises(PlannerFailure):
        OllamaPlanner(settings(), FakeTransport(failure)).plan(spec, snap, assessed)
    assert sim.action_history == []


def test_no_safe_actions_does_not_call_offboard():
    spec, sim, snap, assessed = basic()
    assessed = {}
    tr = FakeTransport()
    with pytest.raises(PlannerFailure):
        OllamaPlanner(settings(), tr).plan(spec, snap, assessed)
    assert tr.posts == []


def test_offline_llm_fallback_returns_safely(tmp_path):
    planner = OllamaPlanner(settings(), FakeTransport('timeout'))
    sim, states, events = run('warehouse', planner=planner, output_dir=tmp_path, verbose=False)
    assert sim._home_return_executed
    assert events[0]['status'] == 'LLM_FAILED_RECOVERY'
    assert events[0]['selected_action_id'] is None
    assert events[0]['rationale'].startswith('PlannerFailure')
    assert (tmp_path/'planner_events.jsonl').is_file()


def test_good_mock_planner_completes_four_missions():
    for mission in ('warehouse', 'exit_sign', 'thermal_inspection', 'mapping'):
        planner = OllamaPlanner(settings(), FakeTransport())
        sim, states, events = run(mission, planner=planner, verbose=False)
        assert sim._home_return_executed, mission
        assert all(g.status.value == 'VERIFIED' for g in states[-1].goals), mission
        assert any(e['status'] == 'LLM_SELECTED' for e in events)
        assert all(e['prompt_bytes'] < 20000 for e in events)


def test_scripted_reference_still_works():
    sim, states, events = run('warehouse', planner_mode='scripted', verbose=False)
    assert sim._home_return_executed
    assert all(g.status.value == 'VERIFIED' for g in states[-1].goals)
    assert any(e['status'] == 'SCRIPTED' for e in events)


def test_env_reads_origin_model_and_temperature(tmp_path, monkeypatch):
    env = tmp_path/'.env'
    env.write_text('OLLAMA_BASE_URL=http://192.168.4.2:11114\nOLLAMA_MODEL=qwen3.5:9b\nOLLAMA_TEMPERATURE=0.2\nOLLAMA_THINK=no\n')
    for name in ('OLLAMA_BASE_URL', 'OLLAMA_MODEL', 'OLLAMA_TEMPERATURE', 'OLLAMA_THINK'):
        monkeypatch.delenv(name, raising=False)
    config = OllamaSettings.from_env(str(env))
    assert config.base_url == 'http://192.168.4.2:11114'
    assert config.model == 'qwen3.5:9b'
    assert config.temperature == 0.2
    assert config.think is False
    monkeypatch.setenv('OLLAMA_MODEL', 'qwen3:8b')
    assert OllamaSettings.from_env(str(env)).model == 'qwen3:8b'


def test_bad_base_url_rejected(monkeypatch):
    monkeypatch.setenv('OLLAMA_BASE_URL', 'ftp://192.0.2.1:11434')
    with pytest.raises(ValueError):
        OllamaSettings.from_env('/does/not/exist')


def test_http_transport_with_loopback_fake_server():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path != '/api/tags':
                self.send_error(404)
                return
            data = {'models': [{'name': 'qwen3:8b'}]}
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            data = json.loads(calls[-1]['messages'][1]['content'])
            action_id = data['allowed_actions'][0]['id']
            body = json.dumps({'message': {'content': json.dumps({
                'action_id': action_id, 'rationale': 'Verified candidate'})}}).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        s = OllamaSettings(f'http://127.0.0.1:{server.server_address[1]}', 'qwen3:8b')
        p = OllamaPlanner(s)
        assert p.check_connection() == ['qwen3:8b']
        spec, sim, snap, assessed = basic()
        result = p.plan(spec, snap, assessed)
        assert result.decision.action_id in {a.id for a in snap.candidates}
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
