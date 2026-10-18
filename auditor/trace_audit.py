#!/usr/bin/env python3
"""Read-only E5-R1 trajectory replay. No dataset loading or efficacy certification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

FREEZE = "834ecf5726e910741df1ca9ff4e674e7c999eed2bfb7adf365040b5421e991d7"
ATOL = RTOL = 1e-10
ANNOTATIONS = {"scope", "interpretation", "sensitivity_scope", "bound_scope", "claim_boundary"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def small(path, limit=8_000_000):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"invalid/oversized file: {path.name}")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("file grew beyond bound")
    return raw


def expect(actual, expected, where="root"):
    """Exact discrete/schema comparisons; explicit tolerance only for real fields."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{where}: expected object")
        extra = set(actual) - set(expected)
        if not extra <= ANNOTATIONS or set(expected) - set(actual):
            raise ValueError(f"{where}: schema mismatch")
        if any(not isinstance(actual[key], str) or len(actual[key]) > 1000 for key in extra):
            raise ValueError(f"{where}: invalid annotation")
        for key, value in expected.items():
            expect(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{where}: sequence mismatch")
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            expect(a, e, f"{where}[{index}]")
    elif type(expected) is float:
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=ATOL, rel_tol=RTOL):
            raise ValueError(f"{where}: numeric replay mismatch")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{where}: discrete replay mismatch")


def dot(x, y):
    if len(x) != len(y):
        raise ValueError("dot-product dimension mismatch")
    return math.fsum(a * b for a, b in zip(x, y, strict=True))


def norm(x):
    return math.sqrt(max(0.0, dot(x, x)))


def centered(x):
    mean = math.fsum(x) / len(x)
    return [v - mean for v in x]


def normalized(x, diagnostic=False):
    c = centered(x)
    n = max(norm(c) if diagnostic else math.hypot(*c), 1e-12)
    result = [v / n for v in c]
    if diagnostic:
        # E5 retains the original native diagnostics arithmetic unchanged.
        return result
    try:
        return vector(result)
    except ValueError:
        # Independent arithmetic replay of E5's conditional repair; no call to
        # model.unit_center or the native predictor supplies these values.
        residuals = [v - x[0] for v in x]
        c = centered(residuals)
        n = max(math.hypot(*c), 1e-12)
        return vector([v / n for v in c])


def vector(x, dimension=None):
    if not isinstance(x, list) or not 2 <= len(x) <= 128 or (dimension is not None and len(x) != dimension):
        raise ValueError("invalid vector dimensions")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in x):
        raise ValueError("nonfinite/non-numeric vector")
    if math.hypot(*x) > 1 + 1e-10 or abs(math.fsum(x) / len(x)) > 1e-10:
        raise ValueError("vector outside centered unit ball")
    return list(x)


def weights(scores):
    high = max(scores)
    e = [math.exp(s - high) for s in scores]
    total = math.fsum(e)
    return [s / total for s in e]


def decode(bank, state):
    s = [dot(m, state) for m in bank]
    i = max(range(len(s)), key=s.__getitem__)
    return {"prediction": i, "winner_index": i, "exact_tie_indices": [j for j, v in enumerate(s) if v == s[i]],
            "near_tie_indices": [j for j, v in enumerate(s) if s[i] - v <= 1e-12], "scores": s}


def decision(bank, state):
    s = [dot(m, state) for m in bank]
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    gap = s[order[0]] - s[order[1]]
    return {"winner": order[0], "margin": gap, "tie": gap == 0,
            "near_tie": gap <= 1e-12 * max(1., *(abs(v) for v in s))}


def energy(bank, z, q, lam):
    s = [6. * dot(m, z) for m in bank]
    h = max(s)
    return .5 * dot(z, z) - (1 - lam) * dot(q, z) - lam / 6 * (h + math.log(math.fsum(math.exp(v-h) for v in s)))


def geometry(bank):
    gram = [[dot(a, b) for b in bank] for a in bank]
    diameter = max(dot([x-y for x, y in zip(a, b, strict=True)], [x-y for x, y in zip(a, b, strict=True)]) for a in bank for b in bank)
    mean = [math.fsum(m[j] for m in bank) / len(bank) for j in range(len(bank[0]))]
    bias = [math.fsum(row) / len(bank) for row in gram]
    return gram, diameter, mean, bias


def native_steps(bank, q, mode="softmax", count=8):
    """Separate implementation preserves original arithmetic order and exact stop."""
    z = list(q)
    rows = []
    for _ in range(count):
        s = [6. * sum(a*b for a, b in zip(z, m, strict=True)) for m in bank]
        ex = [math.exp(max(-700., v-max(s))) for v in s]
        w = [v / sum(ex) for v in ex] if mode == "softmax" else [1. / len(bank)] * len(bank)
        r = [0.] * len(q)
        for wi, m in zip(w, bank, strict=True):
            for j, v in enumerate(m):
                r[j] += wi*v
        nxt = [.6*v + .4*u for v, u in zip(z, r, strict=True)]
        # Clipping at [-128,127] is inactive on this checked convex unit-ball rule.
        delta = sum(abs(a-b) for a, b in zip(z, nxt, strict=True))
        rows.append((z, w, r, nxt, delta))
        z = nxt
        if delta == 0:
            break
    return rows


def anchor_replay(bank, q, alpha=.6, count=8):
    z = list(q)
    history = []
    for t in range(count):
        w = weights([6. * dot(m, z) for m in bank])
        r = [math.fsum(p*m[j] for p, m in zip(w, bank, strict=True)) for j in range(len(q))]
        nxt = [alpha*z[j] + (1-alpha)*(.9*q[j]+.1*r[j]) for j in range(len(q))]
        history.append({"step": t+1, "before": z, "weights": w, "reconstruction": r,
                        "state": nxt, "energy_before": energy(bank, z, q, .1), "energy": energy(bank, nxt, q, .1),
                        "delta_l2": math.dist(z, nxt), "query_distance": math.dist(q, nxt), "decoder": decode(bank, nxt)})
        z = nxt
    return {"state": z, "trace": history, "state_norm": math.hypot(*z), "query_distance": math.dist(q, z),
            "retrieval_score_dot_products": count*len(bank), "decoder_dot_products": len(bank), **decode(bank, z)}


def margin_account(bank, gram, before, w, after, q, lam, winner):
    old = [dot(m, before) for m in bank]
    new = [dot(m, after) for m in bank]
    rival = max((j for j in range(len(bank)) if j != winner), key=lambda j: (new[j], -j))
    differences = [a-b for a, b in zip(gram[winner], gram[rival], strict=True)]
    carry = .6*(old[winner]-old[rival])
    query_term = .4*(1-lam)*(dot(bank[winner], q)-dot(bank[rival], q))
    common = .4*lam*math.fsum(differences)/len(bank)
    selective = .4*lam*math.fsum((wi-1/len(bank))*g for wi, g in zip(w, differences, strict=True))
    observed = new[winner]-new[rival]
    return {"initial_winner": winner, "strongest_after_competitor": rival,
            "retained_state_margin": carry, "fixed_query_margin": query_term,
            "common_memory_margin": common, "selective_weight_margin": selective,
            "observed_after_margin": observed,
            "accounting_residual": observed-carry-query_term-common-selective}


def native_replay(bank, q, gram, diameter):
    d = len(q)
    probes = []
    for candidate in [[1. if j == 0 else -1. if j == 1 else 0. for j in range(d)], [(-1.)**j for j in range(d)]]:
        c = centered(candidate)
        v = [x/norm(c) for x in c]
        if not any(norm([x-y for x, y in zip(v, p, strict=True)]) < 1e-12 for p in probes):
            probes.append(v)
    tangents = [p[:] for p in probes]
    initial = decision(bank, q)
    previous_energy = energy(bank, q, q, 1.)
    traces = []
    contributions = []
    for t, (old, w, r, z, delta) in enumerate(native_steps(bank, q), 1):
        next_tangents = []
        for v in tangents:
            projections = [dot(m, v) for m in bank]
            rv = dot(r, v)
            cv = [math.fsum(p*m[j]*s for p, m, s in zip(w, bank, projections, strict=True))-r[j]*rv for j in range(d)]
            next_tangents.append([.6*x+.4*6*y for x, y in zip(v, cv, strict=True)])
        tangents = next_tangents
        covtrace = max(0., math.fsum(p*dot(m, m) for p, m in zip(w, bank, strict=True))-dot(r, r))
        after = decision(bank, z)
        normalized_after = decision(bank, normalized(z, diagnostic=True))
        e = energy(bank, z, q, 1.)
        dz = [a-b for a, b in zip(z, old, strict=True)]
        a = initial["winner"]
        b = max((j for j in range(len(bank)) if j != a), key=lambda j: (dot(bank[j], z), -j))
        before_scores, after_scores = [dot(m, old) for m in bank], [dot(m, z) for m in bank]
        reconstruction_scores = [math.fsum(g*p for g, p in zip(row, w, strict=True)) for row in gram]
        per_memory = [.4*p*(gram[a][j]-gram[b][j]) for j, p in enumerate(w)]
        carry = .6*(before_scores[a]-before_scores[b])
        observed = after_scores[a]-after_scores[b]
        traces.append({"step": t, "state_norm": norm(z), "centered_state_norm": norm(centered(z)),
                       "near_zero_centered_state": norm(centered(z)) <= 1e-12, "delta_l1": delta,
                       "distance_from_query": norm([a-b for a, b in zip(z, q, strict=True)]),
                       "weight_entropy": -math.fsum(p*math.log(p) for p in w if p > 0),
                       "effective_memory_count": 1./math.fsum(p*p for p in w), "largest_weight": max(w),
                       "covariance_trace": covtrace, "local_update_lipschitz_upper_bound": .6+.4*6.*min(covtrace, diameter/4),
                       "directional_query_jacobian_norms": [norm(v) for v in tangents], "direct_residual_coefficient": .6**t,
                       "raw_decision": after, "normalized_decision": normalized_after,
                       "initial_winner_retained": after["winner"] == a,
                       "normalization_winner_mismatch": after["winner"] != normalized_after["winner"],
                       "energy": e, "energy_change": e-previous_energy,
                       "descent_inequality_residual": e-previous_energy+2.*dot(dz, dz),
                       "initial_winner_margin_budget": {"initial_winner": a, "strongest_after_competitor": b,
                           "retained_state_margin": carry, "per_memory_margin_contributions": per_memory,
                           "summed_memory_margin": math.fsum(per_memory), "observed_after_margin": observed,
                           "margin_accounting_residual": observed-carry-math.fsum(per_memory),
                           "score_recurrence_max_residual": max(abs(x-(.6*y+.4*v)) for x, y, v in zip(after_scores, before_scores, reconstruction_scores, strict=True))}})
        contributions.append(margin_account(bank, gram, old, w, z, q, 1., a))
        previous_energy = e
    return {"initial_decision": initial, "steps_executed": len(traces), "trace": traces,
            "final_state": z, "probe_directions": probes, "final_directional_query_jacobians": tangents}, contributions


def replay_artifacts(model, bounds, anchor, native, query_ids, clean_rows):
    """Pure fixture-testable audit. This function alone cannot certify a real run."""
    expect(model["config"], {"alpha": .6, "beta": 6., "memory_mix": .1, "steps": 8}, "config")
    expect(model["labels"], list(range(len(model["memories"]))), "labels")
    bank = [vector(m) for m in model["memories"]]
    if not 2 <= len(bank) <= 32 or len({len(m) for m in bank}) != 1:
        raise ValueError("invalid replay bank")
    if not isinstance(query_ids, list) or not 1 <= len(query_ids) <= 32 or len(set(query_ids)) != len(query_ids):
        raise ValueError("invalid replay query IDs")
    if set(anchor) != set(query_ids):
        raise ValueError("retained trace query membership mismatch")
    gram, diameter, mean, bias = geometry(bank)
    lip = 6.*diameter/4
    expect(bounds, {"diameter_squared": diameter, "reconstruction_lipschitz_bound": lip,
                    "update_lipschitz_bound": .6+.4*.1*lip, "sufficient_contraction": .6+.4*.1*lip < 1,
                    "strong_convexity_lower_bound": 1-.1*lip, "query_drift_bound": .2*(1-.6**8)}, "bounds")
    trajectories, accounts, expected_clean = [], {}, {}
    for query_id in query_ids:
        saved = anchor[query_id]
        if not isinstance(saved.get("trace"), list) or len(saved["trace"]) != 8:
            raise ValueError("anchor trace must retain eight steps")
        q = vector(saved["trace"][0]["before"], len(bank[0]))
        expected = anchor_replay(bank, q)
        expect(saved, expected, f"anchor.{query_id}")
        nr, terms = native_replay(bank, q, gram, diameter)
        trajectories.append({"id": query_id, **nr})
        a = decode(bank, q)["winner_index"]
        accounts[query_id] = {"native": terms,
            "anchor": [margin_account(bank, gram, row["before"], row["weights"], row["state"], q, .1, a) for row in expected["trace"]]}
        variants = {"direct": decode(bank, q), "direct_repeat8": decode(bank, q),
                    "anchor8": expected, "anchor1": anchor_replay(bank, q, count=1),
                    "anchor_alpha0": anchor_replay(bank, q, alpha=0.)}
        for method, repeats in (("direct", 1), ("direct_repeat8", 8)):
            variants[method].update(retrieval_score_dot_products=0, decoder_dot_products=repeats*len(bank))
        for name, count, mode in [("native_full", 8, "softmax"), ("native_one_step", 1, "softmax"), ("native_uniform", 8, "uniform")]:
            seq = native_steps(bank, q, mode, count)
            final = seq[-1][3]
            variants[name] = {**decode(bank, normalized(final)), "state_norm": math.hypot(*final),
                              "query_distance": math.dist(q, final), "retrieval_score_dot_products": len(seq)*len(bank), "decoder_dot_products": len(bank)}
        for method, value in variants.items():
            expected_clean[(query_id, method)] = {k: v for k, v in value.items() if k not in {"state", "trace", "scores"}}
    counts = {}
    for row in trajectories:
        winner = row["trace"][-1]["raw_decision"]["winner"]
        counts[str(winner)] = counts.get(str(winner), 0)+1
    expect(native, {"status": "COMPLETE_LABEL_FREE_DIAGNOSTIC", "diagnostic_version": 2,
        "config": {"alpha": .6, "beta": 6., "steps": 8, "threshold": 0.},
        "geometry": {"diameter_squared": diameter, "memory_mean_norm": norm(mean),
            "duplicate_memory_count": len(bank)-len({tuple(m) for m in bank}),
            "global_update_lipschitz_upper_bound": .6+.4*6.*diameter/4,
            "global_contraction_sufficient_condition": 6.*diameter/4 < 1},
        "score_feedback_geometry": {"gram_matrix": gram, "uniform_reconstruction_scores": bias,
            "largest_uniform_score_indices": [i for i, v in enumerate(bias) if v == max(bias)],
            "uniform_score_spread": max(bias)-min(bias)}, "query_count": len(query_ids),
        "final_winner_counts": counts, "initial_winner_flip_count": sum(not r["trace"][-1]["initial_winner_retained"] for r in trajectories),
        "trajectories": trajectories}, "native")
    seen = set()
    for row in clean_rows:
        key = row["query_id"], row["method"]
        if key not in expected_clean or key in seen or row["noise"] != 0.:
            raise ValueError("clean replay matrix mismatch")
        seen.add(key)
        expected = expected_clean[key]
        expect({k: row.get(k) for k in expected}, expected, f"clean.{key}")
        if row["initial_prediction"] != expected_clean[(key[0], "direct")]["prediction"]:
            raise ValueError("clean initial decision mismatch")
    if seen != set(expected_clean):
        raise ValueError("incomplete clean replay records")
    return {"queries_replayed": len(query_ids), "clean_method_records_replayed": len(seen),
            "margin_accounts": accounts,
            "maximum_margin_residual": max(abs(t["accounting_residual"]) for r in accounts.values() for seq in r.values() for t in seq),
            "scope": "Internal arithmetic consistency only. Initial-query/data/training provenance, all kNN results, noisy trajectories and independent execution are not replayed. Annotation prose is not certified."}


def load_e5(package):
    package = Path(package).resolve()
    raw = small(package/"FREEZE.json", 100_000)
    if sha(raw) != FREEZE:
        raise ValueError("only the pinned E5-R1 freeze is admitted")
    freeze = strict_json(raw)
    for name, digest in freeze["files"].items():
        if Path(name).name != name or sha(small(package/name, 1_000_000)) != digest:
            raise ValueError("package changed after freeze")
    names = ["safe_io", "epu_diagnostics", "epu_query_anchor", "result_contract", "run_study"]
    for name in names:
        if name in sys.modules and Path(getattr(sys.modules[name], "__file__", "")).resolve() != package/f"{name}.py":
            raise ValueError("ambient package module conflict")
    sys.path.insert(0, str(package))
    try:
        runner = importlib.import_module("run_study")
    finally:
        sys.path.pop(0)
    return runner


def audit_run(package, receipt_path):
    runner = load_e5(package)
    receipt_path = Path(receipt_path).absolute()
    receipt_raw = small(receipt_path, 1_000_000)
    declared = strict_json(receipt_raw)
    if declared.get("status") != "COMPLETE_EXPOSED_DEVELOPMENT" or declared.get("freeze_sha256") != FREEZE:
        raise ValueError("completed pinned E5-R1 run required before payload access")
    admission = runner.diagnostics.resource_admission()
    if admission["status"] != "PASS":
        return {"status": "BLOCKED_RESOURCE", "result_payloads_opened": False, "resource_checks": [admission]}
    receipt, _ = runner.verify_completed_run(receipt_path, FREEZE)
    root = receipt_path.parent
    def read(name):
        raw = small(root/name)
        if sha(raw) != receipt["outputs"][name]:
            raise ValueError(f"artifact changed while reading: {name}")
        return strict_json(raw)
    ids = receipt["query_ids"][:32]
    clean = (r for r in runner.evidence.verified_lines(root/"RAW.jsonl", receipt["outputs"]["RAW.jsonl"])
             if r["query_id"] in ids and r["noise"] == 0. and r["method"] not in {"knn1", "knn3"})
    result = replay_artifacts(read("MODEL.json"), read("BOUNDS.json"), read("ANCHOR_TRACES.json"), read("NATIVE_DIAGNOSTICS.json"), ids, clean)
    completion = runner.diagnostics.resource_admission()
    if completion["status"] != "PASS":
        return {"status": "BLOCKED_RESOURCE", "result_payloads_opened": True,
                "freeze_sha256": FREEZE, "receipt_sha256": sha(receipt_raw),
                "resource_checks": [admission, completion],
                "scope": "Replay completed but its resource completion guard failed; no audit pass is granted."}
    if small(receipt_path, 1_000_000) != receipt_raw or runner.check_freeze()[0] != FREEZE:
        raise ValueError("receipt/package changed during audit")
    # Confirm every artifact still matches the receipt after replay, without reopening data.
    for name, digest in receipt["retained_files"].items():
        if runner.evidence.file_hash(root/name) != digest:
            raise ValueError(f"artifact changed during audit: {name}")
    return {"status": "PASS_INTERNAL_TRACE_REPLAY", "result_payloads_opened": True,
            "freeze_sha256": FREEZE, "receipt_sha256": sha(receipt_raw), "source_outputs": receipt["outputs"],
            "resource_checks": [admission, completion], "audit": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit_run(args.package, args.receipt)
    except (ValueError, KeyError, TypeError, OSError, AttributeError, subprocess.SubprocessError) as exc:
        result = {"status": "REFUSED", "error": str(exc)}
    result.update(at=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(Path(__file__).read_bytes()),
                  new_dataset_opened=False, protected_outcomes_opened=False,
                  claim_boundary="Read-only internal trace audit; no efficacy, novel architecture or independent-reproduction claim.")
    with (args.output/"RECEIPT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.output/"RECEIPT.json")}))
    return 0 if result["status"] == "PASS_INTERNAL_TRACE_REPLAY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
