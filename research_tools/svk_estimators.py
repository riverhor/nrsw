"""Refit the R62 complete estimators and crossed heads from R5 measurements."""

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from nrsw.core import center_residuals, genuine_normalization, pool_aligned
from r29_repro import kernel

if __package__:
    from .svk_evidence_alternatives import fit_diagonal, fit_normal_energy
else:
    from svk_evidence_alternatives import fit_diagonal, fit_normal_energy


NATIVE_METHODS = ("NRSW_COMMON_UNIT", "TAG_DIAGONAL", "GENUINE_ENERGY", "CONCAT_NO_EVIDENCE")
CROSSED_METHODS = ("SD", "DS")
RANKS = (499, 484, 462)
BLOCK = 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Route:
    key: tuple
    evidence: object
    peak_mean: float
    peak_std: float

    def project(self, bags, available, centers=None):
        alpha = np.zeros((len(bags), self.evidence.means.shape[0]), dtype=np.float32)
        strength = np.zeros(len(bags), dtype=np.float64)
        if not available.any():
            return alpha, strength
        centered = None
        if self.evidence.active and centers is not None:
            key = self.evidence.means.tobytes()
            if key not in centers:
                centers[key] = center_residuals(bags[available], self.evidence.means)
            centered = centers[key]
        weights, peaks = project(self.evidence, bags[available], centered)
        alpha[available] = weights
        if self.peak_std > 0:
            strength[available] = (peaks - self.peak_mean) / self.peak_std
        return alpha, strength


def project(evidence, bags, centered):
    if isinstance(evidence, kernel.EvidenceModel):
        if centered is None:
            result = evidence.evaluate(bags)
            return result["alpha"], result["peak"]
        # Same projection and packing as EvidenceModel.evaluate, using a
        # center calculated on these exact rows for these exact means.
        projected = centered @ evidence.factor
        energy = np.sum(projected * projected, axis=2) / evidence.studentizers[None]
        require(np.isfinite(energy).all() and np.all(energy >= 0), "Expected finite nonnegative spectral energy")
        alpha = np.asarray([
            [value / total for value in row] if (total := math.fsum(row))
            else [1 / energy.shape[1]] * energy.shape[1]
            for row in energy.tolist()
        ], dtype=np.float32)
        return alpha, np.max(energy, axis=1)
    result = evidence.evaluate(bags, centered=centered)
    return result["alpha"], result["T"]


def teacher(arm, bags, normal, forged, tags):
    if arm == NATIVE_METHODS[0]:
        return kernel.fit_evidence(bags[normal], bags[forged], tags)
    if arm == NATIVE_METHODS[1]:
        return fit_diagonal(bags[normal], bags[forged], tags)
    return fit_normal_energy(bags[normal])


def fit_routes(training, teacher_cache, *, arms=NATIVE_METHODS[:3]):
    batch = training.features
    bags = batch.local_features
    roles = np.asarray(training.roles)
    normal, forged, bank = (np.flatnonzero(roles == role) for role in ("N", "F", "BANK"))
    bag_hashes = {identity: sha256(bag.tobytes()).digest()
                  for identity, bag in zip(batch.ids, bags, strict=True)}

    def rows(indices):
        return tuple((batch.ids[i], batch.groups[i], bag_hashes[batch.ids[i]]) for i in indices)

    fitted, training_projection = [], []
    for arm in arms:
        require(arm in NATIVE_METHODS[:3], "Expected an evidence-supplying estimator")
        alpha = np.zeros((len(batch), batch.layout.m), dtype=np.float32)
        strength = np.zeros(len(batch), dtype=np.float64)
        routes = []
        for held in (0, 1, None):
            n, f, serve = normal, forged, bank
            if held is not None:
                n = normal[training.folds[normal] != held]
                f = forged[training.folds[forged] != held]
                serve = np.flatnonzero(training.folds == held)
            require(len(n) == (50 if held is None else 25)
                    and len(f) in ((15,) if held is None else (7, 8)),
                    "Expected fixed full and complementary support counts")
            tags = tuple(training.candidates[i] for i in f)
            key = (arm, batch.layout, rows(n))
            if arm != "GENUINE_ENERGY":
                key += (rows(f), tags)
            if key not in teacher_cache:
                teacher_cache[key] = teacher(arm, bags, n, f, tags)
            evidence = teacher_cache[key]
            _, peaks = project(evidence, bags[bank], None)
            mean = float(np.mean(peaks))
            std = float(np.sqrt(np.mean((peaks - mean) ** 2)))
            route = Route(key, evidence, mean, std)
            alpha[serve], strength[serve] = route.project(bags[serve], batch.local_valid[serve])
            routes.append(route)
        fitted.append(tuple(routes))
        training_projection.append((alpha, strength))
    return tuple(fitted), tuple(training_projection)


def fit_model(training, offset, *, crossed=True, teacher_cache=None):
    """Use complementary fits for seeds and the full fit for BANK and queries.

    An optional teacher cache is scoped to one immutable measurement store.
    Every teacher uses a fresh normalizer based on this context's BANK225.
    """
    training.validate()
    batch = training.features
    require(training.roles == ("N",) * 50 + ("F",) * 15 + ("BANK",) * 225
            and batch.local_valid.all(), "Expected complete ordered recovered TRAIN290")
    require(batch.dimensions == (512, 11, 512), "Expected the frozen SVK descriptor dimensions")
    anchor = batch.local_features[:50].astype(np.float64)
    mobile = batch.global_mobile[:50].astype(np.float64)
    delta = anchor.mean(axis=0) - mobile.mean(axis=0)[None]
    _, local = kernel.aligned_parts(batch, delta, offset)
    genuine = training.labels == 0
    mean = local[genuine].astype(np.float64).mean(axis=0).ravel()
    uniform = np.full((len(batch), batch.layout.m), np.float32(1 / batch.layout.m), dtype=np.float32)
    pooled = pool_aligned(batch, uniform, delta, offset)
    _, rms = genuine_normalization(pooled[genuine])
    unit = rms / math.sqrt(pooled.shape[1])
    routes, projections = fit_routes(training, {} if teacher_cache is None else teacher_cache)
    model = Model(batch.layout, frozenset(batch.groups), offset, delta, mean, unit, routes, {}, crossed)
    frames = dict(zip(model.methods, model.prepare(batch).frames(model, projections), strict=True))
    model.heads.update({arm: kernel.fit_head(frame, training.labels) for arm, frame in frames.items()})
    return model, frames


@dataclass
class Prepared:
    batch: kernel.FeatureBatch
    pair: np.ndarray
    local: np.ndarray
    mean_centers: dict
    evidence: dict
    delta: np.ndarray
    offset: np.ndarray

    def score(self, model):
        require(self.batch.layout == model.layout and np.array_equal(self.delta, model.delta)
                and np.array_equal(self.offset, model.offset), "Expected the fitted SVK alignment")
        projections = []
        for complementary in model.routes:
            route = complementary[2]
            # The full teacher's projection can be shared across contexts.
            # The BANK normalization depends on the literal BANK225 roster.
            key = (route.key, route.peak_mean, route.peak_std)
            if key not in self.evidence:
                self.evidence[key] = route.project(self.batch.local_features, self.batch.local_valid,
                                                   self.mean_centers)
            projections.append(self.evidence[key])
        return np.stack([head.score(frame) for head, frame in
                         zip(model.heads.values(), self.frames(model, projections), strict=True)])

    def frames(self, model, projections):
        available = self.batch.local_valid
        dg, m, d = self.batch.dimensions
        ordered = self.local.reshape(len(self.batch), m * d).astype(np.float64)
        ordered[~available] = model.mean
        g, h = self.pair[:, :dg], self.pair[:, dg:]
        pools = [pool_aligned(self.batch, alpha, model.delta, model.offset) for alpha, _ in projections]
        strengths = [model.unit * strength for _, strength in projections]
        for pool, strength in zip(pools, strengths, strict=True):
            yield np.column_stack((g, ordered, pool, strength, h))
        yield np.column_stack((g, ordered, h))
        if model.crossed:
            yield np.column_stack((g, ordered, pools[0], strengths[1], h))
            yield np.column_stack((g, ordered, pools[1], strengths[0], h))


@dataclass
class Model:
    layout: kernel.Layout
    train_groups: frozenset
    offset: np.ndarray
    delta: np.ndarray
    mean: np.ndarray
    unit: float
    routes: tuple
    heads: dict
    crossed: bool

    @property
    def methods(self):
        return NATIVE_METHODS + CROSSED_METHODS if self.crossed else NATIVE_METHODS

    def prepare(self, batch):
        pair, local = kernel.aligned_parts(batch, self.delta, self.offset)
        return Prepared(batch, pair, local, {}, {}, self.delta, self.offset)

    def iter_scores(self, batch, block_size=BLOCK):
        require(not self.train_groups.intersection(batch.groups),
                "CAL or TEST groups overlap the fitted TRAIN groups")
        for first in range(0, len(batch), block_size):
            ix = np.arange(first, min(first + block_size, len(batch)))
            yield self.prepare(batch.subset(ix)).score(self)

    def score(self, batch):
        return np.concatenate(tuple(self.iter_scores(batch)), axis=1)
