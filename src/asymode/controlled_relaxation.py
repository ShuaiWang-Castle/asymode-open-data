"""I20 controlled relaxation inside the host's single damage MLP.

This implements KERNEL_CONTROLLED_RELAXATION_DESIGN_20260928.md: all-subset
geographic landmark features; level/departure controls; four eight-dimensional
states; four rank-two Cayley factors per state; anchored bounded writing; and
bounded readout. It reuses W2/b2 and subclasses GCRKLayer only for the host's
existing kernel-slot protocol. No legacy GCRK parameters/adjoint are inherited.

The positive length-scale floor is 1e-3; order masses each have floor 1e-6.
Initialization uses at most 1024 deterministic unique-county pairs, not labels.
The unchanged warmup is zero at step 0. With alpha=0 the host is exactly recovered;
after step 0 alpha can receive gradients before the internal parameters do.

All scalar/state bounds are architectural. They do not identify physical damage,
guarantee parameter-gradient conditioning, or imply contraction across different
weather paths. Training must pass only fitting geography/county IDs here and
only fitting hidden sequences to calibrate_.
"""
from __future__ import annotations

import bisect
import math
import random
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .gcrk import GCRKLayer, prefix_reference, SCALE_FLOOR, DROP_PATH, WARMUP_STEPS
from .controlled_relaxation_scan import controlled_sequence_adjoint

GEO_DIM = 40
LANDMARKS = 32
MODES, STATE_DIM, PLANES = 4, 8, 4
HIDDEN_DIM = MODES * STATE_DIM
GEO_FEATURES, WEATHER_FEATURES, FUSION_DIM = 72, 64, 32
HEAD_DIM = 56
LENGTH_FLOOR, ORDER_FLOOR = 1e-3, 1e-6
PAIR_BUDGET = 1024
TAU_MIN, TAU_MAX = 2.0, 96.0
TAU_INITIAL = (3.0, 8.0, 24.0, 72.0)
HEAD_SLICES = {"d": slice(0, 32), "tau": slice(32, 36),
               "theta": slice(36, 52), "omega": slice(52, 56)}


def all_subset_kernel(x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor,
                      masses: torch.Tensor) -> torch.Tensor:
    """Normalized all-order product kernel without enumerating subsets.

    x=[N,D], y=[M,D], positive lengths=[D], masses=[D]. The recurrence
    computes mean products of every subset of each order. All operations retain
    autograd, including the landmark coordinates, lengths and order masses.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be [N,D] and [M,D]")
    dim = x.shape[1]
    if lengths.shape != (dim,) or masses.shape != (dim,) or dim == 0:
        raise ValueError("one length and mass are required per dimension/order")
    # Coordinate differences are small compared with the time-state graph.
    base = torch.exp(-0.5 * ((x[:, None] - y[None]) / lengths).square())
    means = base.new_ones((*base.shape[:2], 1))
    for j in range(1, dim + 1):
        orders = torch.arange(j + 1, device=x.device, dtype=x.dtype)
        means = ((j - orders) / j) * F.pad(means, (0, 1)) + \
            (orders / j) * base[..., j - 1, None] * F.pad(means, (1, 0))
    return (means[..., 1:] * masses).sum(-1)


def bounded_vector(v: torch.Tensor) -> torch.Tensor:
    return v / torch.sqrt(1.0 + v.square().sum(-1, keepdim=True))


def cayley_rank2_apply(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                      eta: torch.Tensor) -> torch.Tensor:
    """Apply (I-eta J)^-1(I+eta J)x, J=a b^T-b a^T, via a 2x2 solve.

    Leading dimensions broadcast. The determinant is
    1+eta^2 (||a||^2 ||b||^2-<a,b>^2) >= 1 in exact arithmetic.
    No time-indexed dense matrices are constructed.
    """
    aa, bb, ab = a.square().sum(-1), b.square().sum(-1), (a * b).sum(-1)
    ax, bx = (a * x).sum(-1), (b * x).sum(-1)
    det = 1.0 + eta.square() * (aa * bb - ab.square())
    s1 = ((1.0 + eta * ab) * bx - eta * bb * ax) / det
    s2 = (eta * aa * bx + (1.0 - eta * ab) * ax) / det
    return x + (2.0 * eta)[..., None] * (a * s1[..., None] - b * s2[..., None])


def bounded_readout(p: torch.Tensor) -> torch.Tensor:
    """Operator-norm bound, with the prescribed (possibly nonsmooth) max(1,sigma)."""
    return p / torch.linalg.matrix_norm(p, ord=2).clamp_min(1.0)


def controlled_sequence(deposit: torch.Tensor, rho: torch.Tensor, eta: torch.Tensor,
                        a: torch.Tensor, b: torch.Tensor,
                        initial: torch.Tensor | None = None) -> torch.Tensor:
    """Autograd reference recurrence. Inputs [B,T,4,8], [B,T,4], [B,T,4,4]."""
    state = torch.zeros_like(deposit[:, 0]) if initial is None else initial
    states = []
    for t in range(deposit.shape[1]):
        for r in range(a.shape[1]):
            state = cayley_rank2_apply(state, a[:, r], b[:, r], eta[:, t, :, r])
        state = rho[:, t, :, None] * state + (1.0 - rho[:, t, :, None]) * deposit[:, t]
        states.append(state)
    return torch.stack(states, 1)


def _fit_geography(fit_geo: torch.Tensor, county_ids: Sequence | torch.Tensor | None,
                   seed: int) -> tuple[dict[str, torch.Tensor], dict]:
    """Pure fit-only preprocessing. Work on CPU, with deterministic ID ties."""
    fit = fit_geo.detach().to(device="cpu", dtype=torch.float64)
    if fit.ndim != 2 or fit.shape[1] != GEO_DIM or not len(fit):
        raise ValueError("expected nonempty fitting geography [N,40]")
    if not bool(torch.isfinite(fit).all()):
        raise ValueError("fitting geography must be finite")
    centered = torch.tanh(fit / 3.0)
    center = centered.mean(0)  # I18: county-event rows, not deduplicated counties.
    centered = centered - center
    rms = torch.sqrt(0.1 ** 2 + centered.square().sum(-1).mean())
    transformed = centered / rms
    if county_ids is None:
        # Only a synthetic/API convenience. The registered runner supplies FIPS.
        groups: dict[tuple, int] = {}
        for i, row in enumerate(fit.tolist()):
            groups.setdefault(tuple(row), i)
        unique_rows = [groups[key] for key in sorted(groups)]
        ids = ["synthetic_" + str(i) for i in range(len(unique_rows))]
        id_source = "synthetic_geo_rows"
    else:
        raw_ids = county_ids.detach().cpu().tolist() if isinstance(county_ids, torch.Tensor) else list(county_ids)
        if len(raw_ids) != len(fit):
            raise ValueError("county_ids must align with fitting rows")
        groups = {}
        for i, value in enumerate(raw_ids):
            key = str(value)
            if key in groups and not torch.allclose(fit[i], fit[groups[key]], atol=1e-7, rtol=1e-6):
                raise ValueError(f"inconsistent fitting geography for county {key}")
            groups.setdefault(key, i)
        ids = sorted(groups)
        unique_rows = [groups[key] for key in ids]
        id_source = "provided_county_ids"
    indices = torch.tensor(unique_rows, dtype=torch.long)
    unique = transformed[indices]
    n = len(unique)
    # All ties use the unique counties' stable sorted order.
    first = int((unique - unique.mean(0)).square().sum(-1).argmin())
    chosen = [first]
    distance = (unique - unique[first]).square().sum(-1)
    while len(chosen) < min(LANDMARKS, n):
        distance[chosen] = -1.0
        nxt = int(distance.argmax())
        chosen.append(nxt)
        distance = torch.minimum(distance, (unique - unique[nxt]).square().sum(-1))
    selected = [chosen[j % len(chosen)] for j in range(LANDMARKS)]
    pair_count = n * (n - 1) // 2
    pair_ids = sorted(random.Random(int(seed) + 7).sample(range(pair_count), min(PAIR_BUDGET, pair_count)))
    row_starts = [i * n - i * (i + 1) // 2 for i in range(n)]
    pairs = []
    for pair_id in pair_ids:
        i = bisect.bisect_right(row_starts, pair_id) - 1
        pairs.append((i, i + 1 + pair_id - row_starts[i]))
    diffs = (unique[[i for i, _ in pairs]] - unique[[j for _, j in pairs]]).abs() if pairs else unique.new_empty((0, GEO_DIM))
    lengths, fallbacks = [], []
    for j in range(GEO_DIM):
        values = diffs[:, j][diffs[:, j] > 0]
        if len(values):
            lengths.append(values.median().clamp_min(LENGTH_FLOOR))
        else:
            lengths.append(unique.new_tensor(1.0))
            fallbacks.append(j)
    buffers = dict(geo_center=center, geo_rms_scale=rms,
                   landmarks=unique[selected], landmark_fit_indices=indices[selected],
                   initial_lengths=torch.stack(lengths))
    metadata = dict(county_id_source=id_source, n_fit_rows=len(fit), n_unique_counties=n,
                    landmark_count=LANDMARKS, repeated_landmarks=max(0, LANDMARKS - n),
                    landmark_county_ids=[ids[j] for j in selected],
                    landmark_fit_indices=indices[selected].tolist(),
                    pair_seed=int(seed) + 7, pair_budget=PAIR_BUDGET,
                    lengthscale_pairs=pairs, lengthscale_fallback_columns=fallbacks,
                    length_floor=LENGTH_FLOOR, order_floor=ORDER_FLOOR,
                    preprocessing="I18 county-event tanh center and scalar RMS; unique-county landmarks/pairs")
    return buffers, metadata


class ControlledRelaxationLayer(GCRKLayer):
    """Drop-in kernel-slot layer; fit_geo is standardized *fitting* geography.

    county_ids must be supplied by real training. Missing IDs and repeated
    landmarks are only supported to keep small synthetic/API tests useful.
    checkpoint_steps=0 disables time rematerialization; geo_checkpoint separately
    controls rematerializing the all-subset kernel graph.
    use_adjoint=False selects the ordinary-autograd recurrence, including when
    higher-order derivatives are needed; the exact adjoint supports first order.
    Drop-path is sampled once per training_step, shared across county microbatches.
    """

    def __init__(self, original: nn.Linear, fit_geo: torch.Tensor,
                 county_ids: Sequence | torch.Tensor | None = None,
                 private_seed: int = 1729, checkpoint_steps: int = 16,
                 use_adjoint: bool = True, geo_checkpoint: bool = True):
        nn.Module.__init__(self)
        if not isinstance(original, nn.Linear) or original.in_features != HIDDEN_DIM:
            raise ValueError("controlled relaxation requires a 32-input second damage linear layer")
        if checkpoint_steps < 0:
            raise ValueError("checkpoint_steps must be nonnegative")
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        w = self.weight
        buffers, self.fit_metadata = _fit_geography(fit_geo, county_ids, private_seed)
        for name, value in buffers.items():
            self.register_buffer(name, value.to(device=w.device, dtype=w.dtype) if value.is_floating_point() else value.to(w.device))
        self.register_buffer("scale", w.new_tensor(1.0))
        self.register_buffer("level_scale", w.new_ones(HIDDEN_DIM))
        self.register_buffer("departure_scale", w.new_ones(HIDDEN_DIM))
        self.register_buffer("calibration_step", torch.tensor(-1, device=w.device))
        self.register_buffer("training_step", torch.tensor(0, device=w.device))
        target = (self.initial_lengths - LENGTH_FLOOR).clamp_min(torch.finfo(w.dtype).eps)
        self.length_raw = nn.Parameter(target + torch.log(-torch.expm1(-target)))
        self.order_logits = nn.Parameter(w.new_zeros(GEO_DIM))
        generator = torch.Generator(device="cpu").manual_seed(int(private_seed))

        def normal(*shape, scale=1.0):
            return nn.Parameter((scale * torch.randn(*shape, generator=generator)).to(w))

        self.fusion_geo = normal(FUSION_DIM, GEO_FEATURES, scale=1.0 / math.sqrt(GEO_FEATURES))
        self.fusion_weather = normal(FUSION_DIM, WEATHER_FEATURES, scale=1.0 / math.sqrt(WEATHER_FEATURES))
        self.fusion_bias = nn.Parameter(w.new_zeros(FUSION_DIM))
        self.head_geo = normal(HEAD_DIM, GEO_FEATURES, scale=0.1 / math.sqrt(GEO_FEATURES))
        self.head_weather = normal(HEAD_DIM, WEATHER_FEATURES, scale=0.1 / math.sqrt(WEATHER_FEATURES))
        self.head_fusion = normal(HEAD_DIM, FUSION_DIM, scale=0.1 / math.sqrt(FUSION_DIM))
        self.head_bias = nn.Parameter(w.new_zeros(HEAD_DIM))
        with torch.no_grad():
            for value in (self.head_geo, self.head_weather, self.head_fusion):
                value[HEAD_SLICES["tau"]].zero_()
            prob = (w.new_tensor(TAU_INITIAL).log() - math.log(TAU_MIN)) / math.log(TAU_MAX / TAU_MIN)
            self.head_bias[HEAD_SLICES["tau"]].copy_(torch.logit(prob))
        self.plane_a = normal(MODES, PLANES, STATE_DIM, scale=1.0 / math.sqrt(STATE_DIM))
        self.plane_b = normal(MODES, PLANES, STATE_DIM, scale=1.0 / math.sqrt(STATE_DIM))
        self.readout = nn.Parameter(torch.eye(HIDDEN_DIM, device=w.device, dtype=w.dtype))
        self.alpha = nn.Parameter(w.new_zeros(()))
        self.checkpoint_steps = int(checkpoint_steps)
        self.use_adjoint = bool(use_adjoint)
        self.geo_checkpoint = bool(geo_checkpoint)
        self.bounded_opening, self.space_on = True, False
        self._drop = torch.Generator(device="cpu").manual_seed(int(private_seed) + 1)
        self._drop_step, self._drop_value, self._drop_override = None, 1.0, None
        self.last_mask, self.last_calibration = 1.0, {}

    def get_extra_state(self):
        return dict(fit_metadata=self.fit_metadata, checkpoint_steps=self.checkpoint_steps,
                    use_adjoint=self.use_adjoint, geo_checkpoint=self.geo_checkpoint)

    def set_extra_state(self, state):
        self.fit_metadata = state["fit_metadata"]
        self.checkpoint_steps = int(state.get("checkpoint_steps", self.checkpoint_steps))
        self.use_adjoint = bool(state.get("use_adjoint", self.use_adjoint))
        self.geo_checkpoint = bool(state.get("geo_checkpoint", self.geo_checkpoint))

    def length_scales(self):
        return LENGTH_FLOOR + F.softplus(self.length_raw)

    def order_masses(self):
        return ORDER_FLOOR + (1.0 - GEO_DIM * ORDER_FLOOR) * self.order_logits.softmax(0)

    def normalized_geography(self, g):
        return (torch.tanh(g / 3.0) - self.geo_center) / self.geo_rms_scale

    def geography_features(self, g: torch.Tensor, deduplicate: bool = True):
        """Compute the expensive kernel once per unique geo row, not per hour.

        Geography is normally fixed data. For geo-input gradient diagnostics we
        disable deduplication, since unique() has no suitable input derivative.
        """
        if g.ndim != 2 or g.shape[1] != GEO_DIM:
            raise ValueError("geography must be [B,40]")
        if deduplicate and not g.requires_grad:
            unique, inverse = torch.unique(g, dim=0, return_inverse=True)
        else:
            unique, inverse = g, None
        normalized = self.normalized_geography(unique)
        args = (normalized, self.landmarks, self.length_scales(), self.order_masses())
        if self.geo_checkpoint and torch.is_grad_enabled():
            basis = checkpoint(all_subset_kernel, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            basis = all_subset_kernel(*args)
        features = torch.cat((normalized, basis / math.sqrt(LANDMARKS)), -1)
        return features if inverse is None else features[inverse]

    def code_of(self, g):
        return self.geography_features(g)

    def condition(self, g, u=None):
        """Structured controls, not the legacy (lambda,a,gain) tuple.

        u=[B,T,64] is the already scaled level/departure representation. This
        small public method supports diagnostics and anchored-gradient tests.
        """
        e = self.geography_features(g)
        if u is None:
            u = e.new_zeros((len(e), 1, WEATHER_FEATURES))
        fg, hg, anchor = self._geographic_terms(e)
        return self._controls(u, fg, hg, anchor)

    def _geographic_terms(self, e):
        fg = F.linear(e, self.fusion_geo, self.fusion_bias)
        hg = F.linear(e, self.head_geo, self.head_bias)
        anchor = hg + F.linear(torch.tanh(fg), self.head_fusion)
        return fg, hg, anchor  # anchor stays connected to every geographic parameter.

    def _controls(self, u, fg, hg, anchor):
        psi = torch.tanh(fg[:, None] + F.linear(u, self.fusion_weather))
        # Algebraically hg+B*u+C*psi. Forming the difference before the linear
        # map makes u=0 exactly anchored even when time/batch GEMM shapes differ.
        raw = anchor[:, None] + F.linear(u, self.head_weather) + F.linear(
            psi - torch.tanh(fg)[:, None], self.head_fusion)
        deposit = (torch.tanh(raw[..., :32]) - torch.tanh(anchor[:, None, :32])) / (2.0 * math.sqrt(STATE_DIM))
        tau = torch.exp(math.log(TAU_MIN) + math.log(TAU_MAX / TAU_MIN) *
                        torch.sigmoid(raw[..., HEAD_SLICES["tau"]]))
        return dict(deposit=deposit.reshape(*u.shape[:2], MODES, STATE_DIM), tau=tau,
                    rho=torch.exp(-1.0 / tau),
                    eta=(0.25 * torch.tanh(raw[..., HEAD_SLICES["theta"]])).reshape(*u.shape[:2], MODES, PLANES),
                    gain=torch.exp(math.log(2.0) * torch.tanh(raw[..., HEAD_SLICES["omega"]])))

    @torch.no_grad()
    def calibrate_(self, h: torch.Tensor, step: int) -> dict:
        return self.calibrate_chunks_((h,), step)

    @torch.no_grad()
    def calibrate_chunks_(self, hidden_chunks, step: int) -> dict:
        """Exact full-fit calibration from county chunks, including a short last chunk.

        Each county's entire history remains intact. Coordinate square sums are
        accumulated in float64; only scalar departure norms (excluding t=0) are
        retained for the same lower median used by torch.Tensor.median().
        Buffers are committed only after the whole iterator validates.
        """
        level_sum = torch.zeros(HIDDEN_DIM, dtype=torch.float64)
        departure_sum = torch.zeros_like(level_sum)
        norm_parts = []
        n_counties, n_cells = 0, 0
        for h in hidden_chunks:
            if h.ndim != 3 or h.shape[-1] != HIDDEN_DIM or h.shape[1] < 2 or not len(h):
                raise ValueError("calibration needs nonempty [B,T>=2,32] fitting hidden sequences")
            if not bool(torch.isfinite(h).all()):
                raise ValueError("calibration hidden values must be finite")
            departure = h - prefix_reference(h)
            norm_parts.append(torch.linalg.vector_norm(departure, dim=-1)[:, 1:].reshape(-1).cpu())
            level_sum += h.to(dtype=torch.float64).square().sum((0, 1)).cpu()
            departure_sum += departure.to(dtype=torch.float64).square().sum((0, 1)).cpu()
            n_counties += len(h)
            n_cells += h.shape[0] * h.shape[1]
        if not n_counties:
            raise ValueError("calibration iterator is empty")
        norms = torch.cat(norm_parts)
        scalar = norms.median().clamp_min(SCALE_FLOOR)
        level = (level_sum / n_cells).sqrt().clamp_min(SCALE_FLOOR)
        deviation = (departure_sum / n_cells).sqrt().clamp_min(SCALE_FLOOR)
        if not bool(torch.isfinite(scalar) & torch.isfinite(level).all() & torch.isfinite(deviation).all()):
            raise ValueError("calibration scales must be finite")
        self.scale.copy_(scalar)
        self.level_scale.copy_(level)
        self.departure_scale.copy_(deviation)
        self.calibration_step.fill_(int(step))
        self.last_calibration = dict(step=int(step), scale=float(self.scale),
                                     n_counties=n_counties, n_values=int(norms.numel()),
                                     level_scale=self.level_scale.detach().cpu().tolist(),
                                     departure_scale=self.departure_scale.detach().cpu().tolist())
        return dict(self.last_calibration)

    @torch.no_grad()
    def project_(self):
        """All constraints are enforced in forward, not by in-place projection."""
        return self

    def set_drop_override(self, mask: float | None):
        if mask is not None and float(mask) not in (0.0, 1.0):
            raise ValueError("drop override must be 0, 1, or None")
        self._drop_override = None if mask is None else float(mask)

    def _mask(self):
        if not self.training:
            return 1.0
        if self._drop_override is not None:
            return self._drop_override
        step = int(self.training_step)
        if self._drop_step != step:
            self._drop_step = step
            self._drop_value = float(torch.rand((), generator=self._drop) >= DROP_PATH)
        return self._drop_value

    def _chunk(self, state, h, reference, fg, hg, anchor, a, b, p, diagnostics):
        u = torch.cat((torch.asinh(h / self.level_scale),
                       torch.asinh((h - reference) / self.departure_scale)), -1)
        control = self._controls(u, fg, hg, anchor)
        scan = controlled_sequence_adjoint if self.use_adjoint else controlled_sequence
        states = scan(control["deposit"], control["rho"], control["eta"], a, b, state)
        readout = F.linear((control["gain"][..., None] * states).flatten(-2) / math.sqrt(MODES), p)
        if diagnostics:
            return (states[:, -1], readout, states, control["deposit"], control["tau"],
                    control["rho"], control["eta"], control["gain"])
        return states[:, -1], readout

    def forward(self, h, g, diagnostics=False, exit_open=True, space=None):
        if int(self.calibration_step) < 0:
            raise RuntimeError("calibrate_ on fitting hidden sequences before first forward")
        if h.ndim != 3 or h.shape[-1] != HIDDEN_DIM or h.shape[1] == 0 or len(h) != len(g):
            raise ValueError("expected aligned h=[B,T,32] and g=[B,40]")
        if space is not None:
            raise ValueError("controlled relaxation has no spatial coupling")
        reference = prefix_reference(h)
        e = self.geography_features(g)
        fg, hg, anchor = self._geographic_terms(e)
        a, b = bounded_vector(self.plane_a), bounded_vector(self.plane_b)
        p = bounded_readout(self.readout)
        state = h.new_zeros((len(h), MODES, STATE_DIM))
        size = self.checkpoint_steps or h.shape[1]
        pieces, diagnostic_pieces = [], []
        for start in range(0, h.shape[1], size):
            args = (state, h[:, start:start + size], reference[:, start:start + size], fg, hg, anchor, a, b, p)
            if self.checkpoint_steps and torch.is_grad_enabled():
                def run(*values):
                    return self._chunk(*values, diagnostics=diagnostics)
                result = checkpoint(run, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                result = self._chunk(*args, diagnostics=diagnostics)
            state = result[0]
            pieces.append(result[1])
            if diagnostics:
                diagnostic_pieces.append(result[2:])
        response = torch.cat(pieces, 1)
        beta = torch.tanh(self.alpha)
        self.last_mask = self._mask()
        effect = self.ramp() * beta * self.scale * response
        out = F.linear(h + (self.last_mask if exit_open else 0.0) * effect, self.weight, self.bias)
        if diagnostics:
            names = ("state", "deposit", "tau", "rho", "eta", "mode_gain")
            details = {name: torch.cat([part[j] for part in diagnostic_pieces], 1)
                       for j, name in enumerate(names)}
            details.update(reference=reference, departure=h - reference, response=response,
                           effect=effect, beta=beta, scale=self.scale,
                           level_scale=self.level_scale, departure_scale=self.departure_scale,
                           geography_features=e, length_scales=self.length_scales(),
                           order_masses=self.order_masses(), bounded_readout=p,
                           ramp=h.new_tensor(self.ramp()))
            return out, details
        return out
