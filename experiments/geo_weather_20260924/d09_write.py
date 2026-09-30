"""D09: change only CRK writing, leaving the frozen I20 implementation intact.

The new writing retains all original parameters and nonlinear geography/weather
fusion. Geographic offsets modulate weather-induced writing instead of occupying
one side of the final tanh box. No labels, extra weather MLP or parameter blocks
are introduced. The other controls and the state/readout recurrence are inherited.

For fixed parameters, writing is Lipschitz in calibrated hidden weather u with
constant 2.5/(2*sqrt(8)), and in existing geographic features e with constant
5/(2*sqrt(8)). These are component bounds, not a raw-weather-to-outage guarantee.
They neither guarantee nonzero sensitivity nor identify physical damage.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from asymode.controlled_relaxation import (
    ControlledRelaxationLayer, MODES, STATE_DIM, WEATHER_FEATURES,
)

WEATHER_RADIUS = math.sqrt(WEATHER_FEATURES)
WRITE_DIVISOR = 2.0 * math.sqrt(STATE_DIM)
WEATHER_LIPSCHITZ = 2.5 / WRITE_DIVISOR
GEOGRAPHY_FEATURE_LIPSCHITZ = (0.5 * WEATHER_RADIUS + 1.0) / WRITE_DIVISOR


def spectral_cap(matrix: torch.Tensor) -> torch.Tensor:
    """Exact operator-norm cap; no power-iteration state or fitted scale."""
    return matrix / torch.linalg.matrix_norm(matrix, ord=2).clamp_min(1.0)


def radius_map(vector: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Smooth norm cap with Lipschitz constant one and identity tangent at zero."""
    return vector / torch.sqrt(1.0 + vector.square().sum(-1, keepdim=True) / radius ** 2)


class IncrementalWriteRelaxationLayer(ControlledRelaxationLayer):
    """Fresh initialized CRK with a different, explicitly bounded write function.

    This is not a function-preserving rewrite or a migration of the trained CRK.
    All constructor arguments, parameters, geography and calibration are the old
    CRK's. Only deposit changes; tau/rho/eta/gain are bit-identical for the same
    u/parameters. Full geo40 and all-order geographic features remain accessible.
    """

    def _controls(self, u, fg, hg, anchor):
        controls = super()._controls(u, fg, hg, anchor)
        wd, cd = self.head_weather[:MODES * STATE_DIM], self.head_fusion[:MODES * STATE_DIM]
        gd = self.head_geo[:MODES * STATE_DIM]
        b, c, f = spectral_cap(wd), spectral_cap(cd), spectral_cap(self.fusion_weather)
        # Separate writing-only views: the original controls above are unchanged.
        geo_fusion_norm = torch.linalg.matrix_norm(self.fusion_geo, ord=2)
        fg_write = fg / geo_fusion_norm.clamp_min(1.0)
        anchor_geo_bound = (torch.linalg.matrix_norm(gd, ord=2) +
                            torch.linalg.matrix_norm(cd, ord=2) * geo_fusion_norm).clamp_min(1.0)
        geography_gain = 1.0 + 0.5 * torch.tanh(anchor[..., :MODES * STATE_DIM] / anchor_geo_bound)
        v = radius_map(u, WEATHER_RADIUS)
        nonlinear_increment = (torch.tanh(fg_write[:, None] + F.linear(v, f)) -
                               torch.tanh(fg_write)[:, None])
        raw = (geography_gain[:, None] * F.linear(v, b) +
               F.linear(nonlinear_increment, c)) / WRITE_DIVISOR
        controls['deposit'] = radius_map(raw.reshape(*u.shape[:2], MODES, STATE_DIM))
        return controls

    def get_extra_state(self):
        state = super().get_extra_state()
        state['d09_write_design'] = dict(version=1, weather_radius=WEATHER_RADIUS,
            write_divisor=WRITE_DIVISOR, weather_lipschitz=WEATHER_LIPSCHITZ,
            geography_feature_lipschitz=GEOGRAPHY_FEATURE_LIPSCHITZ,
            extra_learned_parameters=0, other_controls_unchanged=True)
        return state
