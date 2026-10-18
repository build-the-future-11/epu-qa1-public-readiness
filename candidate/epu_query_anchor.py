"""EPU-QA1: fixed-query intervention specified in SCIENTIFIC_SPEC.md.

No efficacy, independent-verification, hardware or architecture-novelty claim.
The original EPU implementation is not modified by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import json
import math
from pathlib import Path

MAX_DIM = 128
MAX_MEMORIES = 32
TOL = 1e-10
MIN_BETA = 1e-12


def finite_number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite real numbers required; booleans are not numbers")
    return float(value)


def raw_vector(values):
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= MAX_DIM:
        raise ValueError("vector must have 2..128 coordinates")
    row = tuple(finite_number(x) for x in values)
    if any(abs(x) > 1_000_000 for x in row):
        raise ValueError("raw coordinate exceeds numerical domain")
    return row


def unit_center(values):
    row = raw_vector(values)
    mean = math.fsum(row) / len(row)
    centered = tuple(x - mean for x in row)
    divisor = max(math.hypot(*centered), 1e-12)
    result = tuple(x / divisor for x in centered)
    try:
        # Keep every already-valid E4 normalization result exactly unchanged.
        return checked_state(result)
    except ValueError:
        # Nearly constant values can lose their small mean-centered signal when
        # subtracting the rounded large mean. Center translated residuals only
        # after that failure; this is not an unconditional numerical rewrite.
        residuals = tuple(x - row[0] for x in row)
        residual_mean = math.fsum(residuals) / len(residuals)
        centered = tuple(x - residual_mean for x in residuals)
        divisor = max(math.hypot(*centered), 1e-12)
        return checked_state(tuple(x / divisor for x in centered))


def checked_state(values, dim=None):
    row = raw_vector(values)
    if dim is not None and len(row) != dim:
        raise ValueError("state dimension mismatch")
    if abs(math.fsum(row) / len(row)) > TOL or math.hypot(*row) > 1 + TOL:
        raise ValueError("centered closed-unit-ball state required")
    return row


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dot-product dimension mismatch")
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def softmax(scores):
    if not scores:
        raise ValueError("empty scores")
    values = [finite_number(x) for x in scores]
    high = max(values)
    weights = [math.exp(x - high) for x in values]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


@dataclass(frozen=True)
class Config:
    alpha: float = 0.6
    beta: float = 6.0
    memory_mix: float = 0.1
    steps: int = 8

    def __post_init__(self):
        for name in ("alpha", "beta", "memory_mix"):
            finite_number(getattr(self, name))
        if not 0 <= self.alpha < 1 or not MIN_BETA <= self.beta <= 100:
            raise ValueError("alpha must be in [0,1), beta in [1e-12,100]")
        if not 0 <= self.memory_mix <= 1:
            raise ValueError("memory_mix must be in [0,1]")
        if type(self.steps) is not int or not 0 <= self.steps <= 64:
            raise ValueError("steps must be an integer in [0,64]")


@dataclass(frozen=True)
class PrototypeModel:
    memories: tuple
    labels: tuple
    config: Config = Config()

    def __post_init__(self):
        if not isinstance(self.memories, (list, tuple)) or not 1 <= len(self.memories) <= MAX_MEMORIES:
            raise ValueError("one to 32 memories required")
        if not isinstance(self.labels, (list, tuple)) or len(self.labels) != len(self.memories):
            raise ValueError("one label per memory required")
        if any(type(label) is not int or label < 0 for label in self.labels):
            raise ValueError("nonnegative integer labels required")
        if not isinstance(self.config, Config):
            raise ValueError("Config required")
        bank = tuple(checked_state(row) for row in self.memories)
        if len({len(row) for row in bank}) != 1:
            raise ValueError("memory dimensions differ")
        object.__setattr__(self, "memories", bank)
        object.__setattr__(self, "labels", tuple(self.labels))

    def scores(self, state):
        state = checked_state(state, len(self.memories[0]))
        return tuple(dot(memory, state) for memory in self.memories)

    def decode(self, state):
        scores = self.scores(state)
        winner = max(range(len(scores)), key=scores.__getitem__)
        exact = [i for i, value in enumerate(scores) if value == scores[winner]]
        near = [i for i, value in enumerate(scores) if scores[winner] - value <= 1e-12]
        return {"prediction": self.labels[winner], "winner_index": winner,
                "exact_tie_indices": exact, "near_tie_indices": near,
                "scores": list(scores)}

    def energy(self, state, query):
        state = checked_state(state, len(self.memories[0]))
        query = checked_state(query, len(self.memories[0]))
        scores = [self.config.beta * dot(memory, state) for memory in self.memories]
        high = max(scores)
        logsum = high + math.log(math.fsum(math.exp(x - high) for x in scores))
        lam = self.config.memory_mix
        return (0.5 * dot(state, state) - (1 - lam) * dot(query, state)
                - lam / self.config.beta * logsum)

    def retrieve(self, query, *, trace=False):
        q = checked_state(query, len(self.memories[0]))
        z = q
        history = []
        a, b, lam = self.config.alpha, self.config.beta, self.config.memory_mix
        for step in range(self.config.steps):
            weights = softmax([b * value for value in self.scores(z)])
            recon = tuple(math.fsum(w * m[j] for w, m in zip(weights, self.memories, strict=True))
                          for j in range(len(q)))
            nxt = tuple(a * z[j] + (1 - a) * ((1 - lam) * q[j] + lam * recon[j])
                        for j in range(len(q)))
            checked_state(nxt, len(q))
            if trace:
                history.append({"step": step + 1, "before": list(z), "weights": list(weights),
                                "reconstruction": list(recon), "state": list(nxt),
                                "energy_before": self.energy(z, q), "energy": self.energy(nxt, q),
                                "delta_l2": math.dist(z, nxt), "query_distance": math.dist(q, nxt),
                                "decoder": self.decode(nxt)})
            z = nxt
        return {"state": list(z), "trace": history, "state_norm": math.hypot(*z),
                "query_distance": math.dist(q, z),
                "retrieval_score_dot_products": self.config.steps * len(self.memories)}

    def predict(self, query, *, trace=False):
        result = self.retrieve(query, trace=trace)
        result.update(self.decode(result["state"]))
        result["decoder_dot_products"] = len(self.memories)
        return result

    def bounds(self):
        diameter2 = max(math.dist(a, b) ** 2 for a in self.memories for b in self.memories)
        lipschitz = self.config.beta * diameter2 / 4
        c = self.config.alpha + (1 - self.config.alpha) * self.config.memory_mix * lipschitz
        return {"diameter_squared": diameter2, "reconstruction_lipschitz_bound": lipschitz,
                "update_lipschitz_bound": c, "sufficient_contraction": c < 1,
                "strong_convexity_lower_bound": 1 - self.config.memory_mix * lipschitz,
                "query_drift_bound": 2 * self.config.memory_mix * (1 - self.config.alpha ** self.config.steps),
                "scope": "Floating evaluation of an exact-arithmetic bound; not an interval certificate or accuracy guarantee."}

    def to_dict(self):
        return {"schema_version": 1, "model": "EPU-QA1", "config": asdict(self.config),
                "memories": [list(m) for m in self.memories], "labels": list(self.labels)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema_version", "model", "config", "memories", "labels"}:
            raise ValueError("unexpected checkpoint schema")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["model"] != "EPU-QA1":
            raise ValueError("unsupported checkpoint")
        if not isinstance(value["config"], dict) or set(value["config"]) != set(asdict(Config())):
            raise ValueError("invalid checkpoint config")
        return cls(value["memories"], value["labels"], Config(**value["config"]))

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as out:
            json.dump(self.to_dict(), out, indent=2, allow_nan=False)
            out.write("\n")


def fit_class_means(rows, labels, config=None):
    if config is None:
        config = Config()
    elif not isinstance(config, Config):
        raise ValueError("Config required")
    if len(rows) != len(labels) or not rows:
        raise ValueError("nonempty paired training rows/labels required")
    if any(type(y) is not int or y < 0 for y in labels):
        raise ValueError("training labels must be nonnegative integers")
    checked = [raw_vector(row) for row in rows]
    if len({len(row) for row in checked}) != 1:
        raise ValueError("training dimensions differ")
    if any(x < 0 or x > 1 for row in checked for x in row):
        raise ValueError("training pixels must be scaled into [0,1]")
    classes = sorted(set(labels))
    memories = []
    for label in classes:
        selected = [row for row, y in zip(checked, labels, strict=True) if y == label]
        memories.append(unit_center([math.fsum(row[j] for row in selected) / len(selected)
                                     for j in range(len(checked[0]))]))
    return PrototypeModel(memories, classes, config)


@dataclass(frozen=True)
class NeighborBank:
    """Immutable, once-validated bank for the fixed k=1/k=3 controls.

    Preparation costs O(nd); each query uses O(nd + n log 3) work and O(3)
    ranking storage. The score, numeric row-index tie break and vote are unchanged.
    """
    training: tuple
    labels: tuple
    row_ids: tuple

    def __post_init__(self):
        if any(not isinstance(values, (list, tuple))
               for values in (self.training, self.labels, self.row_ids)):
            raise ValueError("kNN requires training, labels and IDs as finite sequences")
        if not len(self.training) == len(self.labels) == len(self.row_ids) or len(self.training) < 3:
            raise ValueError("kNN requires at least three aligned training records")
        if any(type(label) is not int or label < 0 for label in self.labels):
            raise ValueError("kNN labels must be nonnegative integers")
        if any(type(row_id) is not int or row_id < 0 for row_id in self.row_ids):
            raise ValueError("kNN row indices must be nonnegative integers")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("duplicate training IDs")
        first = checked_state(self.training[0])
        training = (first, *(checked_state(row, len(first)) for row in self.training[1:]))
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "row_ids", tuple(self.row_ids))

    def __len__(self):
        return len(self.training)

    def predict(self, query):
        q = checked_state(query, len(self.training[0]))
        scored = heapq.nsmallest(3, ((-dot(q, row), row_id, label)
                                   for row, row_id, label in zip(
                                       self.training, self.row_ids, self.labels, strict=True
                                   )))
        votes = {}
        for _, _, label in scored:
            votes[label] = votes.get(label, 0) + 1
        return {"knn1": scored[0][2], "knn3": min(votes, key=lambda y: (-votes[y], y)),
                "score_dot_products": len(self.training)}


def nearest_neighbors(query, training, labels, row_ids):
    """Checked one-query convenience wrapper. Reuse NeighborBank for many queries."""
    return NeighborBank(training, labels, row_ids).predict(query)
