"""D11 pure-array causal-path geometry; no files, outcomes or models are loaded.

All fitting is restricted to explicit fit_indices. The caller supplies physical
weather12, fixed-carrier hidden32, raw current xu42, raw context6 and origin y0.
The 49 common columns contain no geo40. Six conditions have exactly 64 kernel
columns and the same 64 input-only unit/hour landmarks. These are residual-probe
features, not a new outage model or a learning-stability guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

SEED = 20260930
TOTAL_HOURS, ORIGIN, FORECAST_START = 216, 71, 72
HOURS = np.arange(FORECAST_START, TOTAL_HOURS)
WINDOWS = (6, 24, 72)
LANDMARK_HOURS = (72, 96, 120, 144, 168, 192, 215)
BATCH, LANDMARKS = 16, 64
PATH_SCALE_FLOOR, GEO_SCALE_FLOOR, EIGEN_FLOOR = 1e-4, 1e-6, 1e-6
CONDITIONS = tuple(p + '_' + g for p in ('R', 'H')
                   for g in ('shared', 'real', 'permuted'))
WEATHER_NAMES = ('cape','cloud','gust','precip','pressure','rh','snowfall',
                 'soil_moisture','t2m_c','u10','v10','wind_speed')
LOG_COLUMNS = (0,3,6)


def transform_weather12(values):
    """Fixed, causal pointwise weather transform; common xu42 stays raw."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 12:
        raise ValueError('registered weather12 order is required')
    result = values.copy()
    result[..., LOG_COLUMNS] = np.log1p(np.maximum(result[..., LOG_COLUMNS],0.0))
    return result


def _hash(*values) -> int:
    payload = '|'.join(str(v) for v in (SEED,) + values).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest(), 'big')


def _strings(values):
    return np.asarray([str(v) for v in values], dtype=str)


def _indices(values, n):
    out = np.asarray(values, dtype=np.int64)
    if out.ndim != 1 or not len(out) or np.any(out < 0) or np.any(out >= n):
        raise ValueError('indices must be a nonempty one-dimensional in-range array')
    if len(np.unique(out)) != len(out):
        raise ValueError('duplicate indices are not allowed')
    return out


def group_weights(groups, original_w):
    """Equal merged-group mass, original design w within each group; no masks."""
    groups = _strings(groups)
    w = np.asarray(original_w, dtype=np.float64)
    if w.shape != groups.shape or not len(w) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('aligned finite positive original weights are required')
    result = np.empty_like(w)
    unique = np.unique(groups)
    for group in unique:
        rows = groups == group
        result[rows] = w[rows] / w[rows].sum() / len(unique)
    return result


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray
    inactive: np.ndarray
    floor: float

    def apply(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError('finite values with the fitted column count are required')
        return (values - self.mean) / self.scale

    def metadata(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(),
                    inactive=self.inactive.tolist(), floor=self.floor)


def _fit_scaler(batches, columns, floor=PATH_SCALE_FLOOR):
    """Streaming moments; each unit has equal mass across supplied hours."""
    first = np.zeros(columns, dtype=np.float64)
    second = np.zeros(columns, dtype=np.float64)
    mass = 0.0
    for values, unit_weight in batches:
        values = np.asarray(values, dtype=np.float64)
        unit_weight = np.asarray(unit_weight, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != columns or unit_weight.shape != values.shape[:1]:
            raise ValueError('scaler batches must be [unit,time,column] with unit weights')
        if not np.isfinite(values).all() or not np.isfinite(unit_weight).all():
            raise ValueError('scaler fitting inputs must be finite')
        first += unit_weight @ values.mean(axis=1)
        second += unit_weight @ np.square(values).mean(axis=1)
        mass += float(unit_weight.sum())
    if mass <= 0:
        raise ValueError('positive fitting mass is required')
    mean = first / mass
    sd = np.sqrt(np.maximum(0.0, second / mass - mean * mean))
    return Scaler(mean, np.maximum(sd, floor), sd < floor, float(floor))


def descriptor_dimension(channels):
    return 3 * channels + 1 + 3 * ((channels + 1) + (channels + 1) ** 2)


def descriptor_blocks(channels):
    sizes = [3 * channels + 1] + [(channels + 1) + (channels + 1) ** 2] * 3
    edges = np.cumsum([0] + sizes)
    return tuple(slice(int(edges[j]), int(edges[j + 1])) for j in range(4))


def causal_descriptor(values, channel_scaler: Scaler):
    """Forecast q with causal 6/24/72-transition signatures and current age.

    A window L contains L+1 nodes when available: start=max(0,t-L).
    Each full second-order coordinate is retained. Prefix inverse implements
    S2(s,t)=S2(0,t)-S2(0,s)-S1(0,s) tensor S1(s,t).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != TOTAL_HOURS:
        raise ValueError('path must be [unit,216,channel]')
    weather = channel_scaler.apply(values)
    n, _, d = weather.shape
    x = np.empty((n, TOTAL_HOURS, d + 1), dtype=np.float64)
    x[..., 0] = np.arange(TOTAL_HOURS, dtype=np.float64)[None] / 72.0
    x[..., 1:] = weather
    delta = np.zeros_like(x)
    delta[:, 1:] = np.diff(x, axis=1)
    s1 = x - x[:, :1]
    s2 = np.zeros((n, TOTAL_HOURS, d + 1, d + 1), dtype=np.float64)
    for t in range(1, TOTAL_HOURS):
        inc = delta[:, t]
        s2[:, t] = s2[:, t - 1] + np.einsum('ni,nj->nij', s1[:, t - 1], inc) + \
            .5 * np.einsum('ni,nj->nij', inc, inc)
    q = np.empty((n, len(HOURS), descriptor_dimension(d)), dtype=np.float64)
    blocks = descriptor_blocks(d)
    q[..., blocks[0]] = np.concatenate((weather[:, HOURS],
        np.broadcast_to(weather[:, :1], (n, len(HOURS), d)),
        delta[:, HOURS, 1:],
        np.broadcast_to((HOURS / 72.0)[None, :, None], (n, len(HOURS), 1))), axis=-1)
    for block, window in zip(blocks[1:], WINDOWS):
        start = np.maximum(0, HOURS - window)
        first = s1[:, HOURS] - s1[:, start]
        second = s2[:, HOURS] - s2[:, start] - np.einsum(
            'nti,ntj->ntij', s1[:, start], first)
        q[..., block] = np.concatenate((first, second.reshape(n, len(HOURS), -1)), axis=-1)
    return q


def path_kernel(x, y, channels):
    """Fixed block-RMS RBF kernel, length1; q is already FIT-standardized."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dim = descriptor_dimension(channels)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != dim or y.shape[1] != dim:
        raise ValueError('aligned two-dimensional path descriptors are required')
    answer = np.zeros((len(x), len(y)), dtype=np.float64)
    for block, weight in zip(descriptor_blocks(channels), (.5, 1 / 6, 1 / 6, 1 / 6)):
        a, b = x[:, block], y[:, block]
        squared = np.maximum(0.0, np.square(a).sum(1)[:, None] +
            np.square(b).sum(1)[None] - 2 * a @ b.T) / a.shape[1]
        answer += weight * np.exp(-.5 * squared)
    return answer


def all_subset_kernel(x, y):
    """Uniform average of all subset-average orders1..40, O(40^2)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.shape[1] == 0:
        raise ValueError('geography arrays must have matching positive dimension')
    dim = x.shape[1]
    means = np.ones((len(x), len(y), 1), dtype=np.float64)
    for j in range(1, dim + 1):
        base = np.exp(-.5 * np.square(x[:, j - 1, None] - y[None, :, j - 1]))
        orders = np.arange(j + 1, dtype=np.float64)
        old = means
        means = np.zeros((*old.shape[:2], j + 1), dtype=np.float64)
        means[..., :j] += old * ((j - orders[:j]) / j)
        means[..., 1:] += base[..., None] * old * (orders[1:] / j)
    return means[..., 1:].mean(-1)


@dataclass(frozen=True)
class Geography:
    median: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    inactive: np.ndarray
    counties: tuple[str, ...]
    profiles: np.ndarray
    donors: dict[str, str]

    @classmethod
    def fit(cls, values, fips):
        values = np.asarray(values, dtype=np.float64)
        fips = _strings(fips)
        if values.shape != (len(fips), 40) or np.isinf(values).any():
            raise ValueError('FIT geography must be [unit,40], finite or NaN')
        counties = tuple(sorted(set(fips.tolist())))
        unique = []
        for county in counties:
            rows = values[fips == county]
            if not all(np.array_equal(row, rows[0], equal_nan=True) for row in rows[1:]):
                raise ValueError('inconsistent geography for the same FIT county')
            unique.append(rows[0])
        unique = np.stack(unique)
        median = np.asarray([np.median(unique[np.isfinite(unique[:, j]), j])
            if np.isfinite(unique[:, j]).any() else 0.0 for j in range(40)])
        filled = np.where(np.isnan(unique), median, unique)
        center = filled.mean(0)
        sd = filled.std(0)
        inactive = sd < GEO_SCALE_FLOOR
        scale = np.where(inactive, 1.0, np.maximum(sd, GEO_SCALE_FLOOR))
        profiles = (filled - center) / scale
        # A seeded cycle is a derangement for n>=2; n=1 has no such permutation.
        order = sorted(counties, key=lambda c: (_hash('geo-derangement', c), c))
        donors = {county: order[(j + 1) % len(order)] for j, county in enumerate(order)}
        return cls(median, center, scale, inactive, counties, profiles, donors)

    def real(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 40 or np.isinf(values).any():
            raise ValueError('geography must be [unit,40], finite or NaN')
        return (np.where(np.isnan(values), self.median, values) - self.center) / self.scale

    def permuted(self, fips):
        fips = _strings(fips)
        lookup = {county: j for j, county in enumerate(self.counties)}
        donors = [self.donors[c] if c in self.donors else
            self.counties[_hash('unseen-geo-donor', c) % len(self.counties)] for c in fips]
        return self.profiles[[lookup[c] for c in donors]], donors

    def metadata(self):
        return dict(unique_fit_counties=len(self.counties), median=self.median.tolist(),
            center=self.center.tolist(), scale=self.scale.tolist(), inactive=self.inactive.tolist(),
            floor=GEO_SCALE_FLOOR, inactive_scale=1.0, permutation_seed=SEED,
            permutation_rule='seeded FIT county cycle; unseen county hash to FIT donor',
            permutation_fixed_points=sum(c == d for c, d in self.donors.items()),
            singleton_derangement_unavailable=len(self.counties) < 2,
            donor_map=self.donors)


def landmark_tokens(fit_indices, unit_ids, groups):
    """Exactly64 distinct FIT unit/hour tokens, group-balanced and label-free."""
    unit_ids, groups = _strings(unit_ids), _strings(groups)
    fit_indices = _indices(fit_indices, len(unit_ids))
    if len(set(unit_ids[fit_indices])) != len(fit_indices):
        raise ValueError('FIT unit IDs must be unique')
    group_order = sorted(set(groups[fit_indices]), key=lambda g: (_hash('group', g), g))
    rows = {g: sorted(fit_indices[groups[fit_indices] == g].tolist(),
        key=lambda i: (_hash('unit', unit_ids[i]), unit_ids[i])) for g in group_order}
    clocks = {}
    for i in fit_indices:
        first = sorted(LANDMARK_HOURS, key=lambda h: (_hash('clock', unit_ids[i], h), h))
        rest = sorted(set(HOURS.tolist()) - set(LANDMARK_HOURS),
            key=lambda h: (_hash('extra-clock', unit_ids[i], h), h))
        clocks[int(i)] = first + rest
    selected, round_number = [], 0
    while len(selected) < LANDMARKS:
        for group in group_order:
            members = rows[group]
            clock_number, unit_number = divmod(round_number, len(members))
            if clock_number < len(HOURS):
                i = members[unit_number]
                selected.append((i, clocks[i][clock_number]))
                if len(selected) == LANDMARKS:
                    break
        round_number += 1
        if round_number > len(fit_indices) * len(HOURS):
            raise ValueError('insufficient distinct landmark tokens')
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class Nystrom:
    transform: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def fit(cls, gram):
        gram = np.asarray(gram, dtype=np.float64)
        if gram.shape != (LANDMARKS, LANDMARKS) or not np.isfinite(gram).all():
            raise ValueError('finite64 by64 landmark Gram is required')
        gram = .5 * (gram + gram.T)
        values, vectors = np.linalg.eigh(gram)
        if values.min() < -1e-8:
            raise ValueError('landmark kernel is not positive semidefinite')
        return cls(vectors / np.sqrt(np.maximum(values, EIGEN_FLOOR))[None], values)

    def apply(self, cross_kernel):
        return np.asarray(cross_kernel, dtype=np.float64) @ self.transform

    def metadata(self):
        clipped = np.maximum(self.eigenvalues, EIGEN_FLOOR)
        return dict(columns=LANDMARKS, eigen_floor=EIGEN_FLOOR,
            eigenvalues=self.eigenvalues.tolist(), floored_count=int((self.eigenvalues < EIGEN_FLOOR).sum()),
            condition_number=float(clipped.max() / clipped.min()),
            effective_rank_above_floor=int((self.eigenvalues >= EIGEN_FLOOR).sum()))


class FittedFamily:
    """Six fitted feature maps sharing exact FIT tokens and a common49 block."""
    def __init__(self, raw_path, hidden_path, xu, context, y0, geo, fips,
                 unit_ids, groups, original_w, fit_indices):
        self.raw_path, self.hidden_path = np.asarray(raw_path), np.asarray(hidden_path)
        self.xu, self.context, self.y0, self.geo = map(np.asarray, (xu, context, y0, geo))
        self.fips, self.unit_ids, self.groups = map(_strings, (fips, unit_ids, groups))
        self.w = np.asarray(original_w, dtype=np.float64)
        n = len(self.unit_ids)
        for array, shape in ((self.raw_path, (n,216,12)), (self.hidden_path,(n,216,32)),
            (self.xu,(n,216,42)), (self.context,(n,6)), (self.y0,(n,)), (self.geo,(n,40)),
            (self.fips,(n,)), (self.groups,(n,)), (self.w,(n,))):
            if array.shape != shape:
                raise ValueError(f'input shape {array.shape} differs from required {shape}')
        self.fit_indices = _indices(fit_indices, n)
        self.fit_weights = group_weights(self.groups[self.fit_indices], self.w[self.fit_indices])
        self.tokens = landmark_tokens(self.fit_indices, self.unit_ids, self.groups)
        self.geography = Geography.fit(self.geo[self.fit_indices], self.fips[self.fit_indices])
        self.channel_scalers, self.q_scalers, self.landmark_q = {}, {}, {}
        for path_name, path in (('R', self.raw_path), ('H', self.hidden_path)):
            channel = _fit_scaler(((self._path_values(path_name,rows),weight)
                for rows,weight in self._fit_batches()),path.shape[-1])
            self.channel_scalers[path_name] = channel
            q_scaler = _fit_scaler(((causal_descriptor(self._path_values(path_name,rows),channel),weight)
                for rows, weight in self._fit_batches()), descriptor_dimension(path.shape[-1]))
            self.q_scalers[path_name] = q_scaler
            landmarks = np.empty((LANDMARKS, descriptor_dimension(path.shape[-1])), dtype=np.float64)
            unique_rows = np.unique(self.tokens[:, 0])
            for start in range(0, len(unique_rows), BATCH):
                rows = unique_rows[start:start+BATCH]
                q = q_scaler.apply(causal_descriptor(self._path_values(path_name,rows),channel))
                for local, row in enumerate(rows):
                    chosen = np.flatnonzero(self.tokens[:, 0] == row)
                    landmarks[chosen] = q[local, self.tokens[chosen, 1] - FORECAST_START]
            self.landmark_q[path_name] = landmarks
        self.aux_scaler = _fit_scaler(((self._aux(rows), weight) for rows, weight in self._fit_batches()), 49)
        landmark_rows = self.tokens[:, 0]
        self.landmark_geo = {'real': self.geography.real(self.geo[landmark_rows]),
            'permuted': self.geography.permuted(self.fips[landmark_rows])[0]}
        self.bases = {}
        for path_name, channels in (('R',12), ('H',32)):
            kw = path_kernel(self.landmark_q[path_name], self.landmark_q[path_name], channels)
            for geography in ('shared', 'real', 'permuted'):
                kg = 1.0 if geography == 'shared' else all_subset_kernel(
                    self.landmark_geo[geography], self.landmark_geo[geography])
                self.bases[path_name+'_'+geography] = Nystrom.fit(kw * kg)

    def _fit_batches(self):
        for start in range(0, len(self.fit_indices), BATCH):
            yield self.fit_indices[start:start+BATCH], self.fit_weights[start:start+BATCH]

    def _aux(self, indices):
        n = len(indices)
        return np.concatenate((np.asarray(self.xu[indices][:, HOURS], dtype=np.float64),
            np.broadcast_to(self.context[indices, None], (n,len(HOURS),6)),
            np.broadcast_to(self.y0[indices,None,None], (n,len(HOURS),1))), axis=-1)

    def _path_values(self,path_name,indices,reference=False):
        path=self.raw_path if path_name=='R' else self.hidden_path
        values=path[indices]
        if reference:
            values=np.broadcast_to(values[:,ORIGIN:ORIGIN+1],values.shape)
        return transform_weather12(values) if path_name=='R' else values

    def _cross_kernels(self, indices, conditions, reference=False):
        n = len(indices)
        g_cross = {}
        for g in set(c.split('_')[1] for c in conditions):
            if g == 'shared':
                g_cross[g] = 1.0
            else:
                values = self.geography.real(self.geo[indices]) if g == 'real' else \
                    self.geography.permuted(self.fips[indices])[0]
                g_cross[g] = all_subset_kernel(values, self.landmark_geo[g])[:,None,:]
        output = {}
        for p in set(c.split('_')[0] for c in conditions):
            path = self.raw_path if p == 'R' else self.hidden_path
            values=self._path_values(p,indices,reference)
            q = self.q_scalers[p].apply(causal_descriptor(values, self.channel_scalers[p]))
            kw = path_kernel(q.reshape(n*len(HOURS),-1), self.landmark_q[p], path.shape[-1])
            kw = kw.reshape(n,len(HOURS),LANDMARKS)
            for condition in conditions:
                prefix, g = condition.split('_')
                if prefix == p:
                    output[condition] = kw * g_cross[g]
        return output

    def transform(self, indices, conditions=None):
        indices = _indices(indices, len(self.unit_ids))
        conditions = CONDITIONS if conditions is None else tuple(conditions)
        if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(CONDITIONS):
            raise ValueError('unique registered D11 conditions are required')
        result = {condition: np.empty((len(indices),len(HOURS),113), dtype=np.float32)
                  for condition in conditions}
        for start in range(0, len(indices), BATCH):
            rows = indices[start:start+BATCH]
            aux = self.aux_scaler.apply(self._aux(rows))
            crosses = self._cross_kernels(rows, conditions)
            for condition, cross in crosses.items():
                feature = self.bases[condition].apply(cross)
                result[condition][start:start+len(rows)] = np.concatenate((aux,feature), axis=-1)
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError('nonfinite transformed feature')
        return result

    def diagnostics(self, indices):
        """Input-only support; no fitted ridge coefficients or outcomes used."""
        indices = _indices(indices, len(self.unit_ids))
        values = {condition: {} for condition in CONDITIONS}
        for start in range(0,len(indices),BATCH):
            rows = indices[start:start+BATCH]
            for reference in (False,True):
                crosses = self._cross_kernels(rows, CONDITIONS, reference=reference)
                prefix = 'reference_' if reference else 'current_'
                for condition, cross in crosses.items():
                    feature = self.bases[condition].apply(cross)
                    leverage = np.square(feature).sum(-1)
                    total = cross.sum(-1)
                    concentration = np.square(cross).sum(-1) / np.maximum(np.square(total),1e-300)
                    metrics = dict(nystrom_leverage=leverage, nystrom_residual=1.0-leverage,
                        max_landmark_similarity=cross.max(-1), landmark_similarity_sum=total,
                        similarity_concentration=concentration, no_numeric_support=total==0)
                    for key, value in metrics.items():
                        values[condition].setdefault(prefix+key,[]).append(value.reshape(-1))
        for condition in CONDITIONS:
            for key, batches in values[condition].items():
                a = np.concatenate(batches).astype(np.float64)
                values[condition][key] = dict(mean=float(a.mean()), minimum=float(a.min()),
                    q10=float(np.quantile(a,.1)), median=float(np.median(a)),
                    q90=float(np.quantile(a,.9)), maximum=float(a.max()))
        unique = sorted(set(self.fips[indices]))
        _, donors = self.geography.permuted(unique)
        seen = np.asarray([county in self.geography.donors for county in self.fips[indices]])
        design_w = self.w[indices]
        if not np.isfinite(design_w).all() or np.any(design_w<=0):
            raise ValueError('positive finite original w required for support mass diagnostics')
        result = dict(conditions=values, units=len(indices), tokens=len(indices)*len(HOURS),
            reference='each unit hour71 constant channels, same clock; support diagnostic only',
            summary_weighting='unweighted input tokens; no outcome masks',
            permutation=dict(unique_requested_counties=len(unique),
                unseen_requested_counties=sum(c not in self.geography.donors for c in unique),
                seen_cases=int(seen.sum()),unseen_cases=int((~seen).sum()),
                seen_original_w=float(design_w[seen].sum()),
                unseen_original_w=float(design_w[~seen].sum()),
                unseen_original_w_fraction=float(design_w[~seen].sum()/design_w.sum()),
                distinct_donors=len(set(donors)), donor_collisions=len(donors)-len(set(donors)),
                heldout_mapping_is_bijection=False))
        return result

    def metadata(self):
        return dict(schema='d11_features_v1', conditions=list(CONDITIONS),
            dimensions=dict(R_descriptor=583,H_descriptor=3463,common=49,nystrom=64,total=113),
            forecast_hours=HOURS.tolist(), channel_fit_hours=list(range(TOTAL_HOURS)),
            descriptor_fit_hours=HOURS.tolist(), batch=BATCH, seed=SEED,
            weather_names=list(WEATHER_NAMES),weather_log_columns=list(LOG_COLUMNS),
            weather_transform='log1p(max(x,0)) for CAPE/precip/snowfall only; common raw xu42 unchanged',
            clock_scale=72.0, windows=list(WINDOWS), window_unit='elapsed transitions',
            kernel_block_weights=[.5,1/6,1/6,1/6], kernel_lengths=1.0,
            geometry_weighting='equal merged-group mass; original w within group; equal hourly mass',
            reference='unit hour71 constant channels at same clock, diagnostics only',
            output_dtype='float32', arithmetic_dtype='float64',
            fit_unit_ids=self.unit_ids[self.fit_indices].tolist(),
            fit_weights=self.fit_weights.tolist(),
            landmarks=[dict(unit_id=self.unit_ids[row], group=self.groups[row],
                hour=int(hour)) for row,hour in self.tokens],
            landmark_clock_rule='hashed priority72/96/120/144/168/192/215; extra forecast clocks only if needed',
            channel_scalers={p:s.metadata() for p,s in self.channel_scalers.items()},
            descriptor_scalers={p:s.metadata() for p,s in self.q_scalers.items()},
            auxiliary_scaler=self.aux_scaler.metadata(), geography=self.geography.metadata(),
            bases={condition:basis.metadata() for condition,basis in self.bases.items()})


def fit_family(raw_path, hidden_path, xu, context, y0, geo, fips, unit_ids,
               groups, w, fit_indices):
    return FittedFamily(raw_path,hidden_path,xu,context,y0,geo,fips,unit_ids,groups,w,fit_indices)


def run_synthetic_checks():
    """Executable synthetic checks; no external arrays, labels or models."""
    import itertools
    checks = []
    def checked(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    checked('raw_dimension',descriptor_dimension(12)==583)
    checked('hidden_dimension',descriptor_dimension(32)==3463)
    checked('equal_group_mass',np.allclose(group_weights(['a','a','b'],[1,3,10]),[.125,.375,.5]))
    logtoy=np.broadcast_to(np.asarray([-2.,0.,3.])[:,None],(3,12)).copy()
    logtrans=transform_weather12(logtoy)
    checked('weather_log_negative_zero_positive',np.allclose(logtrans[:,LOG_COLUMNS],
        np.broadcast_to([0.,0.,np.log(4.)],(3,3)).T))
    checked('weather_nonlog_columns_unchanged',np.array_equal(logtrans[:,[1,2,4,5,7,8,9,10,11]],
        logtoy[:,[1,2,4,5,7,8,9,10,11]]))
    rng = np.random.default_rng(92)
    a,b = rng.normal(size=(4,4)),rng.normal(size=(3,4))
    base=np.exp(-.5*(a[:,None]-b[None])**2)
    direct=sum(np.mean([np.prod(base[...,list(subset)],axis=-1)
        for subset in itertools.combinations(range(4),order)],axis=0) for order in range(1,5))/4
    checked('subset_recurrence_exact',np.allclose(all_subset_kernel(a,b),direct,atol=1e-14))
    checked('subset_diagonal_one',np.allclose(np.diag(all_subset_kernel(a,a)),1))
    checked('subset_psd',np.linalg.eigvalsh(all_subset_kernel(a,a)).min()>-1e-12)
    channel=Scaler(np.zeros(2),np.ones(2),np.zeros(2,dtype=bool),1e-4)
    path=rng.normal(size=(2,216,2))
    first=causal_descriptor(path,channel)
    changed=path.copy();changed[:,121:]+=100
    second=causal_descriptor(changed,channel)
    checked('no_future_descriptor',np.array_equal(first[:,:49],second[:,:49]))
    checked('absolute_clock_retained',np.allclose(first[0,:,6],HOURS/72))
    # Direct local Chen multiplication checks each first and second coordinate.
    x=np.concatenate(((np.arange(216)/72)[None,:,None].repeat(2,0),path),axis=-1)
    for L,block in zip(WINDOWS,descriptor_blocks(2)[1:]):
        for t in (72,120,215):
            s1=np.zeros((2,3));s2=np.zeros((2,3,3))
            for j in range(t-L+1,t+1):
                dx=x[:,j]-x[:,j-1]
                s2+=np.einsum('ni,nj->nij',s1,dx)+.5*np.einsum('ni,nj->nij',dx,dx)
                s1+=dx
            direct=np.concatenate((s1,s2.reshape(2,-1)),axis=1)
            checked(f'signature_window_{L}_{t}',np.allclose(first[:,t-72,block],direct,atol=2e-12))
    n=8
    raw=rng.normal(size=(n,216,12));hidden=rng.normal(size=(n,216,32))
    xu=rng.normal(size=(n,216,42));context=rng.normal(size=(n,6));y0=rng.uniform(size=n)
    geo=rng.normal(size=(n,40));geo[0,0]=np.nan
    fips=np.asarray([f'c{i}' for i in range(n)]);ids=np.arange(n)
    groups=np.asarray(['a','a','b','b','c','c','d','d'])
    family=fit_family(raw,hidden,xu,context,y0,geo,fips,ids,groups,np.arange(1,n+1),np.arange(6))
    features=family.transform(np.arange(n))
    checked('all_conditions_same113',all(v.shape==(n,144,113) for v in features.values()))
    checked('all_features_finite',all(np.isfinite(v).all() for v in features.values()))
    checked('common49_identical',all(np.array_equal(features['R_shared'][...,:49],v[...,:49]) for v in features.values()))
    checked('landmarks64_unique',len(set(map(tuple,family.tokens)))==64)
    checked('landmarks_fit_only',set(family.tokens[:,0])<=set(range(6)))
    checked('permutation_derangement',all(c!=d for c,d in family.geography.donors.items()))
    perm1,donor1=family.geography.permuted(['c0','unseen','unseen'])
    perm2,donor2=family.geography.permuted(['c0','unseen','unseen'])
    checked('permutation_repeatable',np.array_equal(perm1,perm2) and donor1==donor2)
    checked('unseen_county_same_donor',np.array_equal(perm1[1],perm1[2]))
    geo2=geo.copy();geo2[6:]=1e8
    raw2=raw.copy();raw2[6:]=1e8
    hidden2=hidden.copy();hidden2[6:]=1e8
    xu2=xu.copy();xu2[6:]=1e8
    context2=context.copy();context2[6:]=1e8
    y0_2=y0.copy();y0_2[6:]=1e8
    other=fit_family(raw2,hidden2,xu2,context2,y0_2,geo2,fips,ids,groups,np.arange(1,n+1),np.arange(6))
    checked('held_changes_no_fit_metadata',family.metadata()==other.metadata())
    again=other.transform(np.arange(6))
    checked('held_changes_no_fit_features',all(np.array_equal(features[c][:6],again[c]) for c in CONDITIONS))
    support=family.diagnostics(np.arange(n))
    checked('support_unseen_count',support['permutation']['unseen_requested_counties']==2)
    checked('support_reference_explicit','hour71' in support['reference'])
    for condition in CONDITIONS:
        p,g=condition.split('_')
        gram=path_kernel(family.landmark_q[p],family.landmark_q[p],12 if p=='R' else 32)
        if g!='shared':
            gram*=all_subset_kernel(family.landmark_geo[g],family.landmark_geo[g])
        checked('joint_psd_'+condition,np.linalg.eigvalsh(gram).min()>-1e-9)
        phi=family.bases[condition].apply(gram)
        checked('nystrom_residual_'+condition,np.min(1-np.square(phi).sum(1))>=-1e-8)
    row=np.asarray([6])
    changed_raw=raw.copy();changed_raw[6,120:]+=1000
    saved_raw=family.raw_path
    first_ref=family._cross_kernels(row,CONDITIONS,reference=True)
    family.raw_path=changed_raw
    next_ref=family._cross_kernels(row,CONDITIONS,reference=True)
    family.raw_path=saved_raw
    checked('reference_uses_origin71_only',all(np.array_equal(first_ref[c],next_ref[c]) for c in CONDITIONS))
    singleton=Geography.fit(np.zeros((1,40)),['only'])
    checked('singleton_derangement_reported',singleton.metadata()['singleton_derangement_unavailable'])
    # Three fixed group validations: rebuilding geometry excludes each group.
    for validation_group in ('a','b','c'):
        cv_fit=np.flatnonzero((np.arange(n)<6)&(groups!=validation_group))
        cv=fit_family(raw,hidden,xu,context,y0,geo,fips,ids,groups,np.arange(1,n+1),cv_fit)
        checked('cv_landmarks_exclude_'+validation_group,
            not any(groups[row]==validation_group for row in cv.tokens[:,0]))
        checked('cv_geo_fit_only_'+validation_group,
            set(cv.geography.counties)==set(fips[cv_fit]))
    checked('nystrom64_even_constant',Nystrom.fit(np.ones((64,64))).transform.shape==(64,64))
    checked('positive_fixed_geo_orders',np.isclose(all_subset_kernel(np.zeros((1,40)),np.zeros((1,40)))[0,0],1))
    return dict(passed=True,checks=len(checks),names=checks,external_arrays_read=False)


def self_check():
    return run_synthetic_checks()


if __name__ == '__main__':
    import json
    print(json.dumps(run_synthetic_checks(),ensure_ascii=False))
