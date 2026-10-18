#!/usr/bin/env python3
"""Source-pinned, label-free diagnostics for the frozen EPU rule.

This instruments the existing reference; it implements no successor model.
The CLI reads feature payloads only after manifest and resource admission.
Input-role declarations still require a separately reviewed data export.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import types

from safe_io import contained_path, read_bytes, unique_object

REFERENCE_SHA256 = "0d73134bc7854332b8c5038dbc070d1317d93829b429ae1302eb3091af23d660"
MAX_QUERIES = 32
MAX_MEMORIES = 32
MAX_DIM = 128
CONFIG = {"alpha": 0.6, "beta": 6.0, "steps": 8, "threshold": 0.0}


def strict_json(raw):
    def reject(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(max(0.0, dot(a, a)))


def centered(a):
    mean = math.fsum(a) / len(a)
    return [x - mean for x in a]


def unit_center(a):
    c = centered(a)
    scale = max(norm(c), 1e-12)
    return [x / scale for x in c]


def vector(value, dim=None):
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_DIM:
        raise ValueError("vector dimension outside diagnostic budget")
    if dim is not None and len(value) != dim:
        raise ValueError("dimension mismatch")
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in value):
        raise ValueError("vectors must contain finite numbers, not booleans")
    value = [float(x) for x in value]
    if norm(value) > 1.0 + 1e-10 or abs(math.fsum(value)) > 1e-10:
        raise ValueError("requires centered vectors in the unit ball")
    return value


def load_reference(root):
    root = Path(root).absolute()
    raw = read_bytes(root / "code/epu_reference.py", 32_000, root=root)
    if digest(raw) != REFERENCE_SHA256:
        raise ValueError("frozen reference identity changed; re-audit required")
    # Execute precisely the checked bytes, without a second file read or pycache.
    name = "_epu_diagnostic_frozen_reference"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(root / "code/epu_reference.py"), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.EPUCore


def geometry(memories):
    diameter2 = max(dot([x-y for x, y in zip(a, b)],
                        [x-y for x, y in zip(a, b)])
                    for a in memories for b in memories)
    mean = [math.fsum(m[j] for m in memories) / len(memories)
            for j in range(len(memories[0]))]
    return {"diameter_squared": diameter2, "memory_mean_norm": norm(mean),
            "duplicate_memory_count": len(memories) - len({tuple(m) for m in memories}),
            "global_update_lipschitz_upper_bound": 0.6 + 0.4 * 6.0 * diameter2 / 4.0,
            "global_contraction_sufficient_condition": 6.0 * diameter2 / 4.0 < 1.0,
            "bound_scope": "Exact-arithmetic sufficient upper bound, numerically evaluated; not an interval certificate."}


def score_geometry(memories):
    """Prototype-score feedback geometry, not a measured explanation of errors."""
    gram = [[dot(a, b) for b in memories] for a in memories]
    bias = [math.fsum(row) / len(memories) for row in gram]
    return {"gram_matrix": gram, "uniform_reconstruction_scores": bias,
            "largest_uniform_score_indices": [i for i, value in enumerate(bias) if value == max(bias)],
            "uniform_score_spread": max(bias) - min(bias),
            "interpretation": "Row means equal each prototype's score on the mean memory. They describe uniform-weight bias; nonuniform softmax trajectories must be measured separately."}


def score_margin_budget(memories, gram, info, initial_winner):
    """Decompose a score margin using the frozen update; no labels are used.

    The strongest post-step competitor is selected for descriptive accounting.
    A loss of the initial winner is not a classification error without a label.
    """
    before = [dot(m, info['state']) for m in memories]
    after = [dot(m, info['next']) for m in memories]
    recon_scores = [math.fsum(g*w for g, w in zip(row, info['weights'])) for row in gram]
    predicted = [0.6*s + 0.4*r for s, r in zip(before, recon_scores)]
    competitor = max((i for i in range(len(memories)) if i != initial_winner),
                     key=lambda i: (after[i], -i))
    retained = 0.6 * (before[initial_winner] - before[competitor])
    contributions = [0.4*w*(gram[initial_winner][j] - gram[competitor][j])
                     for j, w in enumerate(info['weights'])]
    memory_margin = math.fsum(contributions)
    observed = after[initial_winner] - after[competitor]
    return {"initial_winner": initial_winner, "strongest_after_competitor": competitor,
            "retained_state_margin": retained, "per_memory_margin_contributions": contributions,
            "summed_memory_margin": memory_margin, "observed_after_margin": observed,
            "margin_accounting_residual": observed - retained - memory_margin,
            "score_recurrence_max_residual": max(abs(a-p) for a, p in zip(after, predicted)),
            "interpretation": "Exact score recurrence evaluated in floating point for the fixed unclipped unit-ball rule. Negative margin means initial-winner displacement, not proven error or causal attribution."}


def covariance_action(memories, weights, reconstruction, direction):
    projection = [dot(m, direction) for m in memories]
    r_projection = dot(reconstruction, direction)
    return [math.fsum(w * m[j] * p for w, m, p in zip(weights, memories, projection))
            - reconstruction[j] * r_projection for j in range(len(direction))]


def probe_directions(query):
    # Two fixed centered probes. They are not a full Jacobian/spectral norm.
    d = len(query)
    candidates = [[1.0 if j == 0 else -1.0 if j == 1 else 0.0 for j in range(d)],
                  [(-1.0) ** j for j in range(d)]]
    result = []
    for candidate in candidates:
        c = centered(candidate)
        scale = norm(c)
        direction = [x / scale for x in c]
        if not any(norm([x-y for x, y in zip(direction, old)]) < 1e-12 for old in result):
            result.append(direction)
    return result


def decision(memories, state):
    scores = [dot(m, state) for m in memories]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    margin = scores[order[0]] - scores[order[1]]
    return {"winner": order[0], "margin": margin,
            "tie": scores[order[0]] == scores[order[1]],
            "near_tie": margin <= 1e-12 * max(1.0, *(abs(s) for s in scores))}


def energy(state, memories):
    scores = [6.0 * dot(state, m) for m in memories]
    maximum = max(scores)
    return dot(state, state) / 2 - (maximum + math.log(math.fsum(math.exp(s-maximum) for s in scores))) / 6.0


def trajectory(core_class, memories, query, gram=None):
    core = core_class(dim=len(query), memories=[m[:] for m in memories], z=query[:],
                      alpha=0.6, beta=6.0, threshold=0.0)
    directions = probe_directions(query)
    tangents = [v[:] for v in directions]
    diameter2 = geometry(memories)["diameter_squared"]
    initial = decision(memories, query)
    if gram is None:
        gram = score_geometry(memories)['gram_matrix']
    previous_energy = energy(query, memories)
    rows = []
    for step in range(1, 9):
        info = core.step()  # The frozen source, not a reimplementation of its update.
        weights, reconstruction = info["weights"], info["recon"]
        tangents = [[0.6*vj + 0.4*6.0*cj for vj, cj in zip(v, covariance_action(memories, weights, reconstruction, v))]
                    for v in tangents]
        state = info["next"]
        trace = max(0.0, math.fsum(w*dot(m, m) for w, m in zip(weights, memories)) - dot(reconstruction, reconstruction))
        after = decision(memories, state)
        normalized = decision(memories, unit_center(state))
        norm_centered = norm(centered(state))
        next_energy = energy(state, memories)
        step_norm2 = dot([x-y for x, y in zip(state, info["state"])],
                         [x-y for x, y in zip(state, info["state"])])
        rows.append({
            "step": step, "state_norm": norm(state), "centered_state_norm": norm_centered,
            "near_zero_centered_state": norm_centered <= 1e-12,
            "delta_l1": info["delta"], "distance_from_query": norm([x-y for x, y in zip(state, query)]),
            "weight_entropy": -math.fsum(w*math.log(w) for w in weights if w > 0),
            "effective_memory_count": 1.0 / math.fsum(w*w for w in weights),
            "largest_weight": max(weights), "covariance_trace": trace,
            "local_update_lipschitz_upper_bound": 0.6 + 0.4*6.0*min(trace, diameter2/4.0),
            "directional_query_jacobian_norms": [norm(v) for v in tangents],
            "direct_residual_coefficient": 0.6**step,
            "raw_decision": after, "normalized_decision": normalized,
            "initial_winner_retained": after["winner"] == initial["winner"],
            "normalization_winner_mismatch": after["winner"] != normalized["winner"],
            "energy": next_energy, "energy_change": next_energy - previous_energy,
            "descent_inequality_residual": next_energy - previous_energy + 2.0*step_norm2,
            "initial_winner_margin_budget": score_margin_budget(memories, gram, info, initial['winner']),
        })
        previous_energy = next_energy
        if core.converged():
            break
    return {"initial_decision": initial, "steps_executed": len(rows), "trace": rows,
            "final_state": core.z[:], "probe_directions": directions,
            "final_directional_query_jacobians": tangents,
            "sensitivity_scope": "Derivative in centered input coordinates for the observed step count; excludes preprocessing and changes in the stopping branch. Probe norms are not the spectral norm or total information retained."}


def analyze(core_class, memories, queries):
    if not isinstance(memories, list) or not 2 <= len(memories) <= MAX_MEMORIES:
        raise ValueError("memory count outside diagnostic budget")
    memories = [vector(m) for m in memories]
    dim = len(memories[0])
    if any(len(m) != dim for m in memories):
        raise ValueError("memory dimension mismatch")
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        raise ValueError("query count outside diagnostic budget")
    results = []
    ids = set()
    score_bank = score_geometry(memories)
    for query in queries:
        if not isinstance(query, dict) or set(query) != {"id", "vector"} or not isinstance(query["id"], str) or not query["id"]:
            raise ValueError("queries require only id and vector; labels are refused")
        if query["id"] in ids:
            raise ValueError("duplicate query id")
        ids.add(query["id"])
        result = trajectory(core_class, memories, vector(query["vector"], dim), score_bank['gram_matrix'])
        results.append({"id": query["id"], **result})
    final_winners = Counter(r["trace"][-1]["raw_decision"]["winner"] for r in results)
    return {"status": "COMPLETE_LABEL_FREE_DIAGNOSTIC", "diagnostic_version": 2, "config": CONFIG,
            "claim_boundary": "Descriptive development features only. No labels, accuracy, causal verdict, successor efficacy, independent reproduction or SOTA claim.",
            "geometry": geometry(memories), "query_count": len(queries),
            "score_feedback_geometry": score_bank,
            "final_winner_counts": dict(sorted(final_winners.items())),
            "initial_winner_flip_count": sum(not r["trace"][-1]["initial_winner_retained"] for r in results),
            "trajectories": results}


def id_set(values, name, maximum=100_000):
    if not isinstance(values, list) or not values or len(values) > maximum:
        raise ValueError(f"invalid {name}")
    if any(not isinstance(x, str) or not x or len(x) > 160 for x in values):
        raise ValueError(f"invalid {name} ids")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name} ids")
    return set(values)


def validate_manifest(manifest, root):
    keys = {"schema_version", "status", "role", "protocol", "payload", "training_ids", "query_ids", "protected_ids"}
    if not isinstance(manifest, dict) or set(manifest) != keys:
        raise ValueError("unexpected manifest schema")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported schema version")
    if manifest["status"] != "ADMITTED" or manifest["role"] != "development_diagnostic":
        raise ValueError("development diagnostic is not admitted")
    training = id_set(manifest["training_ids"], "training")
    queries = id_set(manifest["query_ids"], "query", MAX_QUERIES)
    protected = id_set(manifest["protected_ids"], "protected")
    if training & queries or training & protected or queries & protected:
        raise ValueError("training/query/protected ids overlap")
    for key in ("protocol", "payload"):
        entry = manifest[key]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"invalid {key} binding")
        path = entry["path"]
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("relative contained input paths required")
        contained_path(root, root / path)
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("invalid SHA-256 binding")
    if manifest["protocol"]["path"] == manifest["payload"]["path"]:
        raise ValueError("protocol and feature payload must be separate")
    # Protocol metadata only; the feature payload has not been opened.
    raw = read_bytes(root / manifest["protocol"]["path"], 256_000, root=root)
    if digest(raw) != manifest["protocol"]["sha256"]:
        raise ValueError("development protocol binding changed")
    return queries


def resource_sample():
    if platform.system() != "Darwin":
        raise ValueError("resource admission unavailable on this platform; supply a reviewed platform guard")
    vm = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    page = int(re.search(r"page size of (\d+) bytes", vm).group(1))
    def count(name):
        return int(re.search(r"^" + re.escape(name) + r":\s+(\d+)", vm, re.M).group(1))
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=5)
    match = re.search(r"used = ([\d.]+)([KMG])", swap)
    if match is None:
        raise ValueError("cannot parse swap usage")
    return {"free_plus_speculative_bytes": page*(count("Pages free")+count("Pages speculative")),
            "pageouts": count("Pageouts"),
            "swap_used_bytes": int(float(match.group(1))*{"K":1024,"M":1024**2,"G":1024**3}[match.group(2)])}


def resource_admission(sample=resource_sample, pause=time.sleep):
    first = sample()
    pause(1.0)
    second = sample()
    passed = (min(first["free_plus_speculative_bytes"], second["free_plus_speculative_bytes"]) >= 512*1024**2
              and second["pageouts"] == first["pageouts"]
              and second["swap_used_bytes"] <= first["swap_used_bytes"])
    return {"status": "PASS" if passed else "BLOCKED_RESOURCE", "floor_bytes": 512*1024**2,
            "samples": [first, second]}


def execute(manifest_path, epuroot, check_only=False):
    manifest_path = Path(manifest_path).absolute()
    root = manifest_path.parent
    manifest_raw = read_bytes(manifest_path, 1_000_000, root=root)
    manifest = strict_json(manifest_raw)
    expected_queries = validate_manifest(manifest, root)
    reference = load_reference(epuroot)
    if check_only:
        return {"status": "PASS_METADATA_ONLY", "feature_payload_opened": False,
                "execution_admitted": False, "resource_admission": "NOT_CHECKED"}
    resources = resource_admission()
    if resources["status"] != "PASS":
        return {"status": "BLOCKED_RESOURCE", "feature_payload_opened": False, "resources": resources}
    raw = read_bytes(root / manifest["payload"]["path"], 512_000, root=root)
    if digest(raw) != manifest["payload"]["sha256"]:
        raise ValueError("feature payload binding changed")
    payload = strict_json(raw)
    if not isinstance(payload, dict) or set(payload) != {"memories", "queries", "training_ids"}:
        raise ValueError("unexpected feature payload schema; query labels are forbidden")
    if id_set(payload["training_ids"], "training") != set(manifest["training_ids"]):
        raise ValueError("memory training provenance mismatch")
    if not isinstance(payload["queries"], list) or any(not isinstance(q, dict) or "id" not in q for q in payload["queries"]):
        raise ValueError("invalid queries")
    if {q["id"] for q in payload["queries"]} != expected_queries:
        raise ValueError("query membership mismatch")
    result = analyze(reference, payload["memories"], payload["queries"])
    # Refuse success if metadata changed during the run.
    if read_bytes(manifest_path, 1_000_000, root=root) != manifest_raw:
        raise ValueError("manifest changed during diagnostics")
    validate_manifest(manifest, root)
    result.update({"resources": resources, "reference_sha256": REFERENCE_SHA256,
                   "manifest_sha256": digest(manifest_raw), "payload_sha256": digest(raw),
                   "feature_payload_opened": True, "protected_labels_opened": False,
                   "input_provenance_scope": "Membership declarations checked for internal consistency; source export requires separate provenance review."})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epu-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    # Exclusive run directories preserve all refusals and previous results.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = execute(args.manifest, args.epu_root, args.check_only)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
        result = {"status": "REFUSED", "error": str(exc),
                  "claim_boundary": "No scientific result; preserve this refusal."}
    result["diagnostic_source_sha256"] = digest(read_bytes(Path(__file__), 100_000))
    with (args.output_dir / "RECEIPT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(args.output_dir / "RECEIPT.json")}))
    return 0 if result["status"] in {"PASS_METADATA_ONLY", "COMPLETE_LABEL_FREE_DIAGNOSTIC"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
