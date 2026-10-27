"""Finite last-block adaptation, with cached frozen prefixes and validation gates.

No new labels, test-driven fitting, downloaded car classifiers or production writes.
The final encoder remains compatible with the existing portable runtime.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder, load_make_head, visual_inputs
from tools.cache_semantic_v25 import token
from tools.ev26_helpers import acceptance, calibrate, metrics, references, route

OUT = ROOT / 'models/encoder_adaptation_v27'
CACHE = ROOT / 'cache/encoder_adaptation_v27'
BASE = ROOT / 'models/ev_views_v26/semantic_make.pt'
DATA = ROOT / 'cache/ev_views_v26_data.pt'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def prefix(encoder, x):
    x = encoder.conv1(x).flatten(2).transpose(1, 2)
    x = torch.cat([encoder.class_embedding.expand(x.shape[0], 1, -1), x], 1)
    x = encoder.ln_pre(x + encoder.positional_embedding)
    for block in list(encoder.transformer.resblocks)[:-1]:
        x = block(x)
    return x


def finish(block, ln, proj, x):
    return torch.nn.functional.normalize(ln(block(x)[:, 0]) @ proj, dim=-1)


def prepare(test=False):
    if test:
        assert json.loads((OUT / 'validation.json').read_text())['passed']
    name = 'test' if test else 'development'
    meta_path = CACHE / (name + '.json')
    assert not meta_path.exists(), 'Cache already frozen; do not repeat extraction'
    ck = torch.load(BASE, weights_only=True)
    assert not ck['two_views']
    data = torch.load(DATA, weights_only=True)
    rows = [r for r in data['rows'] if (r['split'] == 'test') == test]
    encoder = GeneralVisionEncoder().eval()
    encoder.load_state_dict(ck['encoder_state'])
    jobs = [(r, kind) for kind in ('whole', 'phone') for r in rows]
    tokens = np.lib.format.open_memmap(CACHE / (name + '.npy'), mode='w+', dtype='float32', shape=(len(jobs), 50, 768))
    start = time.time()
    with torch.inference_mode():
        for first in range(0, len(jobs), 8):
            batch = jobs[first:first + 8]
            x = torch.stack([visual_inputs(Image.fromarray(r['body_rgb'].numpy()), r['body_mask'].numpy(), kind)[0] for r, kind in batch])
            tokens[first:first + len(batch)] = prefix(encoder, x).numpy()
            if first % 160 == 0:
                tokens.flush()
                progress = dict(phase=name, completed=first + len(batch), total=len(jobs), seconds=time.time() - start)
                write('prefix_progress.json', progress)
                print(json.dumps(progress), flush=True)
    tokens.flush()
    meta = []
    for r in rows:
        m = {k: r[k] for k in ('image', 'brand', 'group', 'split', 'historical', 'y')}
        m['owner'] = bool(r.get('owner'))
        m['size'] = list(r['body_rgb'].shape[:2])
        m['feature_tokens'] = {kind: token(r, kind) for kind in ('whole', 'phone')}
        meta.append(m)
    binding = dict(rows=meta, brands=data['brands'], base_sha256=sha(BASE), dataset_sha256=sha(DATA), dtype='float32', test=test, seconds=time.time() - start)
    meta_path.write_text(json.dumps(binding), encoding='utf-8')
    print('PREFIX COMPLETE', name, binding['seconds'], flush=True)


def components(ck):
    encoder = GeneralVisionEncoder().eval()
    encoder.load_state_dict(ck['encoder_state'])
    return copy.deepcopy(encoder.transformer.resblocks[-1]), copy.deepcopy(encoder.ln_post), encoder.proj.detach().clone()


def embed(block, ln, proj, tokens):
    block.eval(); ln.eval()
    result = []
    with torch.inference_mode():
        for first in range(0, len(tokens), 32):
            result.append(finish(block, ln, proj, torch.from_numpy(np.array(tokens[first:first + 32]))))
    return torch.cat(result)


def expanded_indices(rows, split):
    ids = [i for i, r in enumerate(rows) if r['split'] == split]
    return ids + [i + len(rows) for i in ids], [rows[i] for i in ids]


def valid_geometry(rows):
    return torch.tensor([min(round(v * (min(1., 96 / max(r['size'])) if k == 'phone' else 1.)) for v in r['size']) >= 40 for k in ('whole', 'phone') for r in rows])


def make_bank(z, rows):
    n = len(rows); names = sorted({r['group'] for r in rows})
    return dict(keys=z.reshape(2, n, -1), labels=torch.tensor([r['y'] for r in rows]), groups=torch.tensor([names.index(r['group']) for r in rows]), k=8, alpha=.5, beta=50., mode='abstention-only')


def train():
    policy = json.loads((OUT / 'policy.json').read_text())
    with (OUT / 'training_once.json').open('x') as f:
        json.dump(dict(started=time.time(), policy_sha256=sha(OUT / 'policy.json')), f)
    ck = torch.load(BASE, weights_only=True)
    meta = json.loads((CACHE / 'development.json').read_text())
    assert meta['base_sha256'] == sha(BASE) and meta['dataset_sha256'] == sha(DATA)
    rows = meta['rows']; brands = meta['brands']; tokens = np.load(CACHE / 'development.npy', mmap_mode='r')
    ti, train_rows = expanded_indices(rows, 'train'); vi, val = expanded_indices(rows, 'validation')
    assert {r['group'] for r in train_rows}.isdisjoint({r['group'] for r in val})
    assert all(r['split'] in ('train', 'validation') for r in rows)
    assert policy['rounds_max'] == len(policy['encoder_learning_rates']) == 2 and policy['epochs_each'] == 3
    truth = torch.tensor([r['y'] for r in train_rows] * 2)
    vy = torch.tensor([r['y'] for r in val] * 2); valid = valid_geometry(val)
    group_names = sorted({r['group'] for r in val}); groups = torch.tensor([group_names.index(r['group']) for r in val] * 2)
    block, ln, proj = components(ck)
    frozen_z = embed(block, ln, proj, tokens)
    head = load_make_head(ck)
    with torch.inference_mode():
        base_logits = head(frozen_z)
        p = (base_logits[vi] / ck['temperature']).softmax(1)
        ref = references(frozen_z[vi], ck['retrieval'], len(brands))
        base_p, base_a = route(p, ref, ck['thresholds'], ck['retrieval']['fallback_thresholds'], brands, valid)
        baseline = metrics(base_p, base_a, val, brands, policy['priority_makes'])
        raw_base = metrics(p, torch.zeros(len(vi), dtype=torch.bool), val, brands, policy['priority_makes'])
    # Verify that the cached split encoder has exactly the old pixel semantics.
    old_features = torch.load(ROOT / 'cache/ev_views_v26_features.pt', weights_only=True)['items']
    expected = torch.stack([torch.nn.functional.normalize(old_features[r['feature_tokens'][k]][0], dim=-1) for k in ('whole', 'phone') for r in rows])
    parity = float((expected - frozen_z).abs().max())
    assert parity < 1e-4, ('Frozen-prefix parity failed', parity)
    del old_features, expected
    sw = torch.tensor([5. if r['owner'] else 1. for r in train_rows] * 2)
    cw = torch.bincount(truth, minlength=len(brands)).float().clamp_min(1).pow(-.2); cw /= cw.mean()
    rounds = []; best = None; start = time.time()
    for rid, lr in enumerate(policy['encoder_learning_rates'], 1):
        torch.manual_seed(270000 + rid)
        block, ln, proj = components(ck); head = load_make_head(ck)
        for param in ln.parameters(): param.requires_grad_(False)
        opt = torch.optim.AdamW([dict(params=block.parameters(), lr=lr), dict(params=head.parameters(), lr=3e-5)], weight_decay=.01)
        chosen = None; history = []
        for epoch in range(policy['epochs_each']):
            block.train(); head.train()
            for ids in torch.randperm(len(ti)).split(32):
                indexes = torch.tensor(ti)[ids]; x = torch.from_numpy(np.array(tokens[indexes.numpy()]))
                opt.zero_grad(); z = finish(block, ln, proj, x); logits = head(z)
                ce = torch.nn.functional.cross_entropy(logits, truth[ids], weight=cw, label_smoothing=.01, reduction='none')
                loss = (ce * sw[ids]).sum() / sw[ids].sum() + .2 * torch.nn.functional.kl_div((logits / 2).log_softmax(1), (base_logits[indexes] / 2).softmax(1), reduction='batchmean') * 4
                loss.backward(); torch.nn.utils.clip_grad_norm_(list(block.parameters()) + list(head.parameters()), 1.); opt.step()
            z = embed(block, ln, proj, tokens); head.eval()
            with torch.inference_mode():
                p = head(z[vi]).softmax(1)
                m = metrics(p, torch.zeros(len(vi), dtype=torch.bool), val, brands, policy['priority_makes'])
                guard = all(m[k]['historical'] >= raw_base[k]['historical'] - .01 for k in m)
                score = .45 * m['whole']['macro'] + .2 * m['phone']['macro'] + .35 * m['whole']['priority_macro']
            record = dict(epoch=epoch + 1, guard=guard, score=score, metrics=m); history.append(record)
            if guard and (chosen is None or score > chosen['score']):
                chosen = dict(score=score, epoch=epoch + 1, block=copy.deepcopy(block.state_dict()), head=copy.deepcopy(head.state_dict()))
            print('FIT', rid, 'EPOCH', epoch + 1, 'SCORE', score, 'GUARD', guard, 'SECONDS', round(time.time() - start), flush=True)
            write('training_progress.json', dict(rounds=rounds, active_round=rid, history=history))
        result = dict(round=rid, learning_rate=lr, history=history, passed=False)
        if chosen:
            block.load_state_dict(chosen['block']); head.load_state_dict(chosen['head']); z = embed(block, ln, proj, tokens)
            with torch.inference_mode():
                logits = head(z[vi]); temp = min((.5, .75, 1., 1.5, 2., 3.), key=lambda t: float(torch.nn.functional.cross_entropy(logits / t, vy)))
                p = (logits / temp).softmax(1); thresholds = calibrate(p, vy, groups, brands, valid, .9)
                bank = make_bank(z[ti], train_rows); ref = references(z[vi], bank, len(brands))
                primary = acceptance(p, thresholds, brands, valid)
                fallback = calibrate(p * .5 + ref * .5, vy, groups, brands, valid, .95, primary)
                rp, ra = route(p, ref, thresholds, fallback, brands, valid); m = metrics(rp, ra, val, brands, policy['priority_makes'])
                checks = dict(whole=m['whole']['top1'] >= baseline['whole']['top1'], phone=m['phone']['top1'] >= baseline['phone']['top1'], historical=all(m[k]['historical'] >= baseline[k]['historical'] - .01 for k in m), priority=m['whole']['priority_macro'] >= baseline['whole']['priority_macro'] + .01, precision=m['whole']['precision'] >= .95, correct_confirmed=m['whole']['correct_confirmed'] >= baseline['whole']['correct_confirmed'], mandatory=all(thresholds[b] < 1 or fallback[b] < 1 for b in policy['mandatory_makes']))
            result.update(selected_epoch=chosen['epoch'], metrics=m, checks=checks, passed=all(checks.values()))
            if result['passed'] and (best is None or chosen['score'] > best['score']):
                bank['fallback_thresholds'] = fallback
                best = dict(chosen, round=rid, temperature=temp, thresholds=thresholds, bank=bank)
        rounds.append(result); write('training_progress.json', dict(rounds=rounds))
    if best:
        candidate = dict(ck, head=best['head'], temperature=best['temperature'], thresholds=best['thresholds'], retrieval=best['bank'])
        state = dict(ck['encoder_state'])
        state.update({'transformer.resblocks.11.' + k: v for k, v in best['block'].items()})
        candidate['encoder_state'] = state
        candidate['adaptation'] = dict(method='own-supervised-last-block-v27', selected_round=best['round'], selected_epoch=best['epoch'], dataset_sha256=sha(DATA))
        torch.save(candidate, OUT / 'semantic_make.pt')
    write('validation.json', dict(passed=best is not None, rounds=rounds, baseline=baseline, prefix_parity_max_error=parity, selected_round=best['round'] if best else None, seconds=time.time() - start, test_used=False))
    if best is None:
        (OUT / 'NO_CHANGE.txt').write_text('Both finite encoder adaptation candidates failed validation. Keep 2.6. No test extraction or production change.\n', encoding='utf-8')
    print('FINITE TRAINING COMPLETE', bool(best), flush=True)


def evaluate():
    assert json.loads((OUT / 'validation.json').read_text())['passed']
    with (OUT / 'evaluation_once.json').open('x') as f: json.dump(dict(started=time.time()), f)
    meta = json.loads((CACHE / 'test.json').read_text()); rows = meta['rows']; brands = meta['brands']
    tokens = np.load(CACHE / 'test.npy', mmap_mode='r'); valid = valid_geometry(rows); policy = json.loads((OUT / 'policy.json').read_text()); results = {}
    for name, path in [('baseline', BASE), ('candidate', OUT / 'semantic_make.pt')]:
        ck = torch.load(path, weights_only=True); block, ln, proj = components(ck); z = embed(block, ln, proj, tokens)
        with torch.inference_mode():
            p = (load_make_head(ck)(z) / ck['temperature']).softmax(1); ref = references(z, ck['retrieval'], len(brands))
            p, a = route(p, ref, ck['thresholds'], ck['retrieval']['fallback_thresholds'], brands, valid)
            results[name] = metrics(p, a, rows, brands, policy['priority_makes'])
    a = results['candidate']; b = results['baseline']
    checks = dict(whole=a['whole']['top1'] >= b['whole']['top1'], phone=a['phone']['top1'] >= b['phone']['top1'], historical=a['whole']['historical'] >= b['whole']['historical'] - .01, precision=a['whole']['precision'] >= .95, correct_confirmed=a['whole']['correct_confirmed'] >= b['whole']['correct_confirmed'], priority=a['whole']['priority_macro'] >= b['whole']['priority_macro'])
    write('evaluation.json', dict(passed=all(checks.values()), checks=checks, results=results, limitations='Reused public heldout set; not an independent Uzbekistan phone exam. Known seven user errors excluded from fitting and selection.'))
    print(json.dumps(dict(checks=checks, results=results)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'train', 'prepare-test', 'evaluate']); args = parser.parse_args()
    torch.set_num_threads(3); OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    if args.phase == 'prepare': prepare()
    elif args.phase == 'prepare-test': prepare(True)
    elif args.phase == 'train': train()
    else: evaluate()
