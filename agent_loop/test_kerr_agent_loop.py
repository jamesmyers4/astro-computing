import json
import re
import numpy as np
import requests
from scipy.optimize import fsolve, brentq
M = 1.0
a = 0.9
r0 = 15.0
TARGET_R_MIN = 4.50
B_CLIP_LOW = 0.1
B_CLIP_HIGH = 15.0
PENALTY_CAPTURED = 50.0
N_ITERATIONS = 4
OLLAMA_MODEL = 'qwen2.5-coder:3b'
OLLAMA_URL = 'http://localhost:11434/api/chat'
def R_photon(r, b):
    return r**4 + (a**2 - b**2) * r**2 + 2 * M * (b - a)**2 * r
def dR_photon_dr(r, b):
    return 4 * r**3 + 2 * (a**2 - b**2) * r + 2 * M * (b - a)**2
def photon_sphere_eqs(vars):
    r, b = vars
    return [R_photon(r, b), dR_photon_dr(r, b)]
r_ph_pro, b_crit_pro = fsolve(photon_sphere_eqs, [2.0, 3.0])
def r_min_of_b(b):
    rs = np.linspace(1.5, r0, 6000)
    Rs = R_photon(rs, b)
    sign_changes = np.where(np.diff(np.sign(Rs)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    idx = sign_changes[-1]
    return brentq(lambda r: R_photon(r, b), rs[idx], rs[idx + 1])
def score_of_b(b):
    r_min = r_min_of_b(b)
    if r_min is None:
        return None, True, PENALTY_CAPTURED
    return r_min, False, abs(r_min - TARGET_R_MIN)
def build_tool_schema():
    return [{
        'type': 'function',
        'function': {
            'name': 'propose_impact_parameter',
            'description': 'Propose the next impact parameter b (units of M) for the photon flyby.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'b': {'type': 'number', 'description': 'impact parameter in units of M'},
                    'reasoning': {'type': 'string', 'description': 'one sentence of reasoning'}
                },
                'required': ['b', 'reasoning']
            }
        }
    }]
def build_messages(history):
    system_prompt = (
        f'You are tuning the impact parameter b (units of black hole mass M) of a photon flying past a '
        f'spinning (Kerr) black hole on a prograde equatorial orbit, to hit a target closest-approach '
        f'radius. Known physics: the closest-approach radius r_min increases monotonically as b increases. '
        f'The critical impact parameter is b_crit = {b_crit_pro:.4f} M; if b <= b_crit the photon is '
        f'captured (no turning point, r_min undefined) and incurs a large penalty score of {PENALTY_CAPTURED}. '
        f'Target closest-approach radius: r_min = {TARGET_R_MIN} M. Score is |r_min - target|, lower is '
        f'better, 0 is a perfect hit. Given the search history below (each entry: the b that was tried, '
        f'the resulting r_min or "captured", and the score), call propose_impact_parameter with the next '
        f'b to try that minimizes the score.'
    )
    user_prompt = 'History so far:\n' + json.dumps(history, indent=2) + '\n\nPropose the next b.'
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
def extract_b_from_content(content):
    try:
        obj = json.loads(content)
        args = obj.get('arguments', obj)
        return float(args['b']), args.get('reasoning', '')
    except Exception:
        pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            args = obj.get('arguments', obj)
            return float(args['b']), args.get('reasoning', '')
        except Exception:
            pass
    match = re.search(r'"?b"?\s*[:=]\s*(-?\d+\.?\d*)', content)
    if match:
        return float(match.group(1)), ''
    return None, None
def fallback_next_b(history):
    valid = [h for h in history if not h['captured']]
    if not valid:
        return (b_crit_pro + B_CLIP_HIGH) / 2.0
    best = min(valid, key=lambda h: h['score'])
    if best['r_min'] < TARGET_R_MIN:
        return best['b'] * 1.15
    return best['b'] * 0.87
def decide_next_b(history):
    current = history[-1]
    messages = build_messages(history)
    tools = build_tool_schema()
    llm_b = None
    request_error = None
    try:
        resp = requests.post(OLLAMA_URL, json={
            'model': OLLAMA_MODEL,
            'messages': messages,
            'tools': tools,
            'stream': False,
            'options': {'temperature': 0.4}
        }, timeout=120)
        resp.raise_for_status()
        message = resp.json()['message']
        tool_calls = message.get('tool_calls')
        if tool_calls:
            args = tool_calls[0]['function']['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            llm_b = float(args['b'])
        else:
            llm_b, _ = extract_b_from_content(message.get('content', ''))
    except Exception as exc:
        request_error = str(exc)
    candidates = [('status_quo', current['b'], current['score'])]
    if llm_b is not None:
        llm_b_clipped = min(max(llm_b, B_CLIP_LOW), B_CLIP_HIGH)
        _, _, llm_score = score_of_b(llm_b_clipped)
        candidates.append(('llm', llm_b_clipped, llm_score))
    numeric_b = min(max(fallback_next_b(history), B_CLIP_LOW), B_CLIP_HIGH)
    _, _, numeric_score = score_of_b(numeric_b)
    candidates.append(('numeric_fallback', numeric_b, numeric_score))
    chosen_source, chosen_b, chosen_score = min(candidates, key=lambda c: c[2])
    meta = {'chosen_source': chosen_source, 'request_error': request_error}
    return chosen_b, meta
def test_agent_loop_improves_on_deliberately_bad_start():
    history = []
    b = 10.0
    for i in range(1, N_ITERATIONS + 1):
        r_min, captured, score = score_of_b(b)
        entry = {'iteration': i, 'b': b, 'r_min': r_min, 'captured': captured, 'score': score}
        if i < N_ITERATIONS:
            b_next, meta = decide_next_b(history + [entry])
            entry.update(meta)
            b = b_next
        history.append(entry)
    assert len(history) == N_ITERATIONS
    assert all(B_CLIP_LOW <= h['b'] <= B_CLIP_HIGH for h in history)
    initial_score = history[0]['score']
    assert initial_score > 1.0
    best = min(history, key=lambda h: h['score'])
    assert best['score'] < initial_score
    assert not best['captured']
