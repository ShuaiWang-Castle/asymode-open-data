"""Local geo-weather mechanisms integrated over a county's customers (experiments/geo_weather_20260924/DESIGN.md).

Input per county-event: node weather [B, K, T, C] in physical units (channels NODE_CH), node attributes
[B, K, A] (ATTR), node weights rho [B, K] summing to one. Output: M county-level mechanism intensities
[B, T, M], Lambda_m(t) = log1p( sum_k rho_k I_m,k(t) ), for the hours t >= t_out (zeros before; the
states still run over every hour).

Every parameter is a shared scalar with a bounded physical range (sigmoid-scaled), initialised at a
literature value; no parameter belongs to a county. Mechanisms:
  0 wind        wind loading softplus((G_k - theta_w) / 2) 2 x (1 + a_can canopy_k),
                node gust G_k = G_cell exp(kappa tpi_k / 100) (exposed terrain is faster)
  1 windthrow   wind loading x soil saturation sigmoid((A_k - a_star) / 10); A_k antecedent liquid
                precipitation with time constant tau_a (1 + wet_gain wet_k) (poorly drained soils drain slower)
  2 snow_ice    accreted load L_k, dL/dt = wet-snow rate + freezing-rain rate - shed(T_k) L_k,
                x (1 + c_can canopy_k) x (1 + G_k / 10)
  3 wet_snow    instantaneous wet-snow rate (solid precipitation inside the ISO 12494 wet-snow window of air
                temperature, 0..+3 C, with soft edges) x (1 + c_can canopy_k)
  4 convective  log1p(CAPE / 1000 x precip) x (1 + a_can canopy_k)
Temperature is moved to the node by a lapse rate: T_k = T_cell + Gamma dz_k / 1000 (Gamma ~ -6.5 K/km),
the dew point by its own lapse rate (~ -2 K/km), and the precipitation phase follows the node's wet-bulb
temperature (Stull 2011) from the moved temperature and humidity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

NODE_CH = ["t2m_c", "d2m_c", "precip", "gust", "cape"]
ATTR = ["dz", "z", "tpi", "slope", "canopy", "forest", "developed", "wet", "area_share"]
MECH = ["wind", "windthrow", "snow_ice", "wet_snow", "convective"]


class _LinRec(torch.autograd.Function):
    """y_0 = x_0; y_s = g_s y_{s-1} + x_s along the last axis (rows independent)."""

    @staticmethod
    def forward(ctx, x, g):
        y = torch.empty_like(x)
        y[..., 0] = x[..., 0]
        for s in range(1, x.shape[-1]):
            y[..., s] = g[..., s] * y[..., s - 1] + x[..., s]
        ctx.save_for_backward(g, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        g, y = ctx.saved_tensors
        gx = torch.empty_like(gy)
        gg = torch.zeros_like(g)
        acc = gy[..., -1].clone()
        for s in range(gy.shape[-1] - 1, 0, -1):
            gx[..., s] = acc
            gg[..., s] = acc * y[..., s - 1]
            acc = gy[..., s - 1] + acc * g[..., s]
        gx[..., 0] = acc
        return gx, gg


linear_recurrence = _LinRec.apply


def linear_recurrence_loop(x, g):
    out = [x[..., 0]]
    for s in range(1, x.shape[-1]):
        out.append(g[..., s] * out[-1] + x[..., s])
    return torch.stack(out, -1)


class Bounded(nn.Module):
    """A scalar in (lo, hi), initialised at init."""

    def __init__(self, lo: float, hi: float, init: float):
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)
        p = (init - lo) / (hi - lo)
        self.raw = nn.Parameter(torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))

    def forward(self):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw)


def wet_bulb(t, rh):
    """Stull (2011) wet-bulb temperature, t in C, rh in %."""
    rh = rh.clamp(1.0, 100.0)
    return (t * torch.atan(0.151977 * torch.sqrt(rh + 8.313659)) + torch.atan(t + rh) - torch.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * torch.atan(0.023101 * rh) - 4.686035)


def rel_humidity(t, td):
    a, b = 17.625, 243.04                                        # Magnus, as in asymode.weather
    return 100.0 * torch.exp(a * td / (b + td) - a * t / (b + t))


class LocalMechanisms(nn.Module):
    def __init__(self, t_out: int = 72, use_dz: bool = True, use_tpi: bool = True, use_canopy: bool = True,
                 use_wet: bool = True):
        super().__init__()
        self.t_out = int(t_out)
        self.use = dict(dz=use_dz, tpi=use_tpi, canopy=use_canopy, wet=use_wet)
        self.gamma = Bounded(-9.8, -3.0, -6.5)            # temperature lapse rate, K / km
        self.gamma_d = Bounded(-6.0, 0.0, -2.0)           # dew-point lapse rate, K / km
        self.kappa = Bounded(0.0, 0.5, 0.1)               # gust speed-up per 100 m of TPI
        self.theta_w = Bounded(8.0, 25.0, 15.0)           # wind-loading threshold, m/s
        self.a_can = Bounded(0.0, 3.0, 0.5)               # canopy amplification of wind loading
        self.c_can = Bounded(0.0, 3.0, 0.5)               # canopy amplification of snow / ice load
        self.snow_t = Bounded(-1.0, 3.0, 1.0)             # wet-bulb threshold of the snow / rain split, C
        self.t_lo = Bounded(-1.0, 1.0, 0.0)               # lower edge of the wet-snow window, C (ISO 12494)
        self.t_hi = Bounded(2.0, 4.0, 3.0)                # upper edge, C (ISO 12494)
        self.w_t = Bounded(0.1, 1.0, 0.3)                 # edge softness, C
        self.tau_a = Bounded(12.0, 240.0, 72.0)           # antecedent-rain time constant, h
        self.wet_gain = Bounded(0.0, 3.0, 1.0)            # longer memory on poorly drained soils
        self.a_star = Bounded(5.0, 150.0, 40.0)           # saturation midpoint, mm
        self.shed_cold = Bounded(0.005, 0.2, 0.02)        # load shedding rate when cold, 1/h
        self.shed_warm = Bounded(0.05, 1.0, 0.3)          # when warm, 1/h

    def attributes(self, attr: torch.Tensor):
        ia = {a: i for i, a in enumerate(ATTR)}
        z = torch.zeros_like(attr[..., 0])
        dz = attr[..., ia["dz"]] if self.use["dz"] else z
        tpi = attr[..., ia["tpi"]] if self.use["tpi"] else z
        can = attr[..., ia["canopy"]] / 100.0 if self.use["canopy"] else z
        wet = attr[..., ia["wet"]] if self.use["wet"] else z
        return dz, tpi, can, wet

    def forward(self, w: torch.Tensor, attr: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        B, K, T, _ = w.shape
        ix = {c: i for i, c in enumerate(NODE_CH)}
        dz, tpi, can, wet = self.attributes(attr)
        dzk = dz[..., None] / 1000.0
        t = w[..., ix["t2m_c"]] + self.gamma() * dzk
        td = torch.minimum(w[..., ix["d2m_c"]] + self.gamma_d() * dzk, t)
        tw = wet_bulb(t, rel_humidity(t, td))
        p = w[..., ix["precip"]].clamp_min(0.0)
        snow_frac = torch.sigmoid((self.snow_t() - tw) / 0.5)
        p_sol = p * snow_frac
        p_liq = p - p_sol
        # states over every hour
        tau_a = self.tau_a() * (1.0 + self.wet_gain() * wet)                        # [B, K]
        keep_a = torch.exp(-1.0 / tau_a)[..., None].expand(B, K, T)
        A = linear_recurrence(p_liq, keep_a)
        band = torch.sigmoid((t - self.t_lo()) / self.w_t()) * torch.sigmoid((self.t_hi() - t) / self.w_t())
        r_ws = p_sol * band
        r_fz = p_liq * torch.sigmoid(-t / 0.5)
        shed = self.shed_cold() + (self.shed_warm() - self.shed_cold()) * torch.sigmoid((t - 1.0) / 0.5)
        L = linear_recurrence(r_ws + r_fz, torch.exp(-shed))
        # intensities for the output hours only
        o = slice(self.t_out, T)
        g = w[:, :, o, ix["gust"]] * torch.exp(self.kappa() * tpi[..., None] / 100.0)
        load_w = torch.nn.functional.softplus((g - self.theta_w()) / 2.0) * 2.0
        veg_w = 1.0 + self.a_can() * can[..., None]
        can_ice = 1.0 + self.c_can() * can[..., None]
        m_wind = load_w * veg_w
        m_wt = m_wind * torch.sigmoid((A[:, :, o] - self.a_star()) / 10.0)
        m_si = L[:, :, o] * can_ice * (1.0 + g / 10.0)
        m_wsn = r_ws[:, :, o] * can_ice
        m_cv = torch.log1p(w[:, :, o, ix["cape"]].clamp_min(0.0) / 1000.0 * p[:, :, o]) * veg_w
        I = torch.stack([m_wind, m_wt, m_si, m_wsn, m_cv], -1)                      # [B, K, T_o, M]
        lam = torch.log1p(torch.einsum("bk,bktm->btm", rho, I))
        return torch.cat([torch.zeros(B, self.t_out, lam.shape[-1], dtype=lam.dtype), lam], 1)

    def values(self) -> dict:
        return {n: float(m()) for n, m in self.named_children() if isinstance(m, Bounded)}
